import contextlib
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

import para_attn
import para_attn.ops as para_attn_ops
import para_attn.primitives as DP
from para_attn.sparse_attn import (
    focus_attn_func,
    FocusAttnMode,
    sparse_kv_attn_func,
    SparseKVAttnMode,
    struct_sparse_attn_func,
    StructSparseAttnMode,
)

try:
    from torch.distributed.tensor.experimental._attention import _templated_ring_attention
except ImportError:
    _templated_ring_attention = None

try:
    import torch.distributed.tensor.experimental._attention as torch_ring_attention
except ImportError:
    torch_ring_attention = None

__all__ = [
    "UnifiedAttnMode",
    "RingAttnMode",
    "UlyssesAttnMode",
    "InBatchAttnMode",
    "SparseKVAttnMode",
    "StructSparseAttnMode",
    "FocusAttnMode",
    "ring_attn_func",
    "ulysses_attn_func",
    "in_batch_attn_func",
    "sparse_kv_attn_func",
    "struct_sparse_attn_func",
    "focus_attn_func",
]


def _sdpa_all_to_all_single(x, mesh):
    if x.requires_grad:
        x = DP.all_to_all_single_autograd_sync(x, output_split_sizes=None, input_split_sizes=None, group=mesh)
    else:
        x = DP.all_to_all_single_sync(x, output_split_sizes=None, input_split_sizes=None, group=mesh)
    return x


def _sdpa_input_all_to_all(x, mesh):
    world_size = DP.get_world_size(mesh)
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x, mesh)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _sdpa_output_all_to_all(x, mesh):
    world_size = DP.get_world_size(mesh)
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x, mesh)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def ulysses_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    mesh=None,
):
    assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
    assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
    assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)

    if mesh is None:
        mesh = DP.get_group()

    query = _sdpa_input_all_to_all(query, mesh)
    key = _sdpa_input_all_to_all(key, mesh)
    value = _sdpa_input_all_to_all(value, mesh)

    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )

    out = _sdpa_output_all_to_all(out, mesh)
    return out


class RingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        mesh,
    ):
        assert _templated_ring_attention is not None, "RingAttnFunc requires a newer version of PyTorch"
        assert torch_ring_attention is not None, "RingAttnFunc requires a newer version of PyTorch"

        with unittest.mock.patch.object(
            torch_ring_attention,
            "_convert_to_f32",
            not para_attn.config.attention.allow_reduced_precision_compute,
            create=True,
        ):
            out, lse = _templated_ring_attention(
                mesh,
                para_attn_ops.attention_forward_with_lse,
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        out = out.to(query.dtype)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for RingAttnFunc is not implemented")


def ring_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    mesh=None,
):
    pg = DP.get_group(mesh)
    world_size = DP.get_world_size(pg)
    if world_size <= 1:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    assert attn_mask is None, "attn_mask is not supported in ring_attn_func when world_size > 1"

    return RingAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        pg,
    )


def in_batch_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
    assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
    assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)
    assert attn_mask is None, "attn_mask is not supported in in_batch_attn_func"
    assert not is_causal, "is_causal is not supported in in_batch_attn_func"

    b, h, s_q, d_qk = query.shape
    _, _, s_kv, d_v = value.shape

    query = query.permute(1, 0, 2, 3).reshape(1, h, b * s_q, -1)
    key = key.permute(1, 0, 2, 3).reshape(1, h, b * s_kv, -1)
    value = value.permute(1, 0, 2, 3).reshape(1, h, b * s_kv, -1)

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    out = out.reshape(h, b, s_q, -1).permute(1, 0, 2, 3)
    return out


def _get_arg(args, kwargs, *field):
    if len(field) == 1:
        if isinstance(field, int):
            if field < len(args):
                return args[field]
            else:
                return None
        else:
            return kwargs.get(field[0])
    else:
        index, name = field
        if index < len(args):
            return args[index]
        else:
            return kwargs.get(name)


def _get_args(args, kwargs, *names):
    results = []
    for i, name in enumerate(names):
        results.append(_get_arg(args, kwargs, i, name))
    return results


class RingAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if RingAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return ring_attn_func(*args, **kwargs, mesh=self._mesh)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UlyssesAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if UlyssesAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return ulysses_attn_func(*args, **kwargs, mesh=self._mesh)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UnifiedAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()

        self._parallel_method = "ulysses"

        if mesh is None:
            self._ulysses_mesh = DP.get_default_group()
            self._ring_mesh = None
        else:
            if isinstance(mesh, dist.ProcessGroup):
                self._ulysses_mesh = mesh
                self._ring_mesh = None
            else:
                assert isinstance(mesh, dist.DeviceMesh), "mesh must be a ProcessGroup or DeviceMesh"

                if "ulysses" in mesh.mesh_dim_names:
                    self._ulysses_mesh = mesh["ulysses"]
                else:
                    self._ulysses_mesh = None
                if "ring" in mesh.mesh_dim_names:
                    self._ring_mesh = mesh["ring"]
                else:
                    self._ring_mesh = None

                assert (
                    self._ulysses_mesh is not None or self._ring_mesh is not None
                ), "mesh must have ulysses or ring dim"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if UnifiedAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            parallel_method = self._parallel_method
            if parallel_method == "ulysses":
                with self._set_parallel_method("ring"), self:
                    if self._ulysses_mesh is None:
                        return func(*args, **kwargs)
                    return ulysses_attn_func(*args, **kwargs, mesh=self._ulysses_mesh)
            elif parallel_method == "ring":
                with self._set_parallel_method("none"), self:
                    if self._ring_mesh is None:
                        return func(*args, **kwargs)
                    return ring_attn_func(*args, **kwargs, mesh=self._ring_mesh)
            elif parallel_method == "none":
                if para_attn.config.attention.force_dispatch_to_custom_ops:
                    return para_attn_ops.attention_forward(*args, **kwargs)
                return func(*args, **kwargs)
            else:
                raise ValueError(f"Unknown parallel method: {parallel_method}")

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled

    @contextlib.contextmanager
    def _set_parallel_method(self, method):
        old_parallel_method = self._parallel_method
        self._parallel_method = method
        try:
            yield
        finally:
            self._parallel_method = old_parallel_method


class InBatchAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if InBatchAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return in_batch_attn_func(*args, **kwargs)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled
