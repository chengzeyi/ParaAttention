import contextlib

import torch
import torch.distributed as dist
import torch.nn.functional as F

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
from para_attn.utils import _get_force_dispatch_to_custom_ops, _torch_version_check, BaseTorchFunctionMode

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


@torch.compiler.assume_constant_result
def _setup_ring_attn_func_forward():
    _convert_to_f32 = None
    _cp_options_convert_to_f32 = None
    _cp_options_enable_load_balance = None
    _cp_options_rotate_method = None

    assert torch_ring_attention is not None, "RingAttnFunc requires a newer version of PyTorch"
    _convert_to_f32 = getattr(torch_ring_attention, "_convert_to_f32", None)
    if _convert_to_f32 is not None:
        torch_ring_attention._convert_to_f32 = not para_attn.config.attention.allow_reduced_precision_compute
    _cp_options = getattr(torch_ring_attention, "_cp_options", None)
    if _cp_options is not None:
        _cp_options_convert_to_f32 = getattr(_cp_options, "convert_to_f32", None)
        if _cp_options_convert_to_f32 is not None:
            _cp_options.convert_to_f32 = not para_attn.config.attention.allow_reduced_precision_compute
        _cp_options_enable_load_balance = getattr(_cp_options, "enable_load_balance", None)
        if _cp_options_enable_load_balance is not None:
            _cp_options.enable_load_balance = False
        _cp_options_rotate_method = getattr(_cp_options, "rotate_method", None)
        if _cp_options_rotate_method is not None:
            _cp_options_rotate_method = _cp_options_rotate_method.value
            _cp_options.rotate_method = torch_ring_attention._RotateMethod.ALL_TO_ALL
    return _convert_to_f32, _cp_options_convert_to_f32, _cp_options_enable_load_balance, _cp_options_rotate_method


@torch.compiler.assume_constant_result
def _cleanup_ring_attn_func_forward(
    _convert_to_f32, _cp_options_convert_to_f32, _cp_options_enable_load_balance, _cp_options_rotate_method
):
    if _convert_to_f32 is not None:
        torch_ring_attention._convert_to_f32 = _convert_to_f32
    if _cp_options_convert_to_f32 is not None:
        torch_ring_attention._cp_options.convert_to_f32 = _cp_options_convert_to_f32
    if _cp_options_enable_load_balance is not None:
        torch_ring_attention._cp_options.enable_load_balance = _cp_options_enable_load_balance
    if _cp_options_rotate_method is not None:
        torch_ring_attention._cp_options.rotate_method = torch_ring_attention._RotateMethod(_cp_options_rotate_method)


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

        seq_dim_args = []
        if _torch_version_check("__ge__", "2.6.0"):
            seq_dim_args = [query.ndim - 2]
        settings = _setup_ring_attn_func_forward()
        try:
            out, lse = _templated_ring_attention(
                mesh,
                *seq_dim_args,
                para_attn_ops.attention_forward_with_lse,
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        finally:
            _cleanup_ring_attn_func_forward(*settings)
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


class RingAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if RingAttnMode.disabled:
            return super().__torch_function__(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return self._call_ring_attn_func(*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _call_ring_attn_func(self, *args, **kwargs):
        mesh = self._mesh
        return ring_attn_func(*args, **kwargs, mesh=mesh)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UlyssesAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if UlyssesAttnMode.disabled:
            return super().__torch_function__(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return self._call_ulysses_attn_func(*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _call_ulysses_attn_func(self, *args, **kwargs):
        mesh = self._mesh
        return ulysses_attn_func(*args, **kwargs, mesh=mesh)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UnifiedAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
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
            return super().__torch_function__(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return self._call_unified_attn_func(*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _call_unified_attn_func(self, *args, **kwargs):
        func = F.scaled_dot_product_attention
        parallel_method = self._parallel_method
        if parallel_method == "ulysses":
            self._parallel_method = "ring"
            try:
                with self:
                    if self._ulysses_mesh is None:
                        out = func(*args, **kwargs)
                    else:
                        out = ulysses_attn_func(*args, **kwargs, mesh=self._ulysses_mesh)
            finally:
                self._parallel_method = parallel_method
        elif parallel_method == "ring":
            self._parallel_method = "none"
            try:
                with self:
                    if self._ring_mesh is None:
                        out = func(*args, **kwargs)
                    else:
                        out = ring_attn_func(*args, **kwargs, mesh=self._ring_mesh)
            finally:
                self._parallel_method = parallel_method
        elif parallel_method == "none":
            if _get_force_dispatch_to_custom_ops():
                out = para_attn_ops.attention_forward(*args, **kwargs)
            else:
                out = func(*args, **kwargs)
        else:
            raise ValueError(f"Unknown parallel method: {parallel_method}")

        return out

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class InBatchAttnMode(BaseTorchFunctionMode):
    disabled = False

    @torch.compiler.disable
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if InBatchAttnMode.disabled:
            return super().__torch_function__(func, types, args, kwargs)

        if func is F.scaled_dot_product_attention:
            return in_batch_attn_func(*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled
