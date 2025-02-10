import pytest

import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface
from para_attn.distributed.mp_runner import MPDistRunner


class ParallelAttnRunner(MPDistRunner):
    @property
    def world_size(self):
        return torch.cuda.device_count()

    @property
    def attn_func(self):
        return None

    def attn_mode(self, device):
        return None

    def _test_attn_func_task(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        func = self.attn_func
        if func is None:
            return

        if compile:
            func = torch.compile(func)

        if is_causal and S_Q != S_KV:
            return

        with torch.no_grad(), torch.cuda.device(self.rank):
            torch.manual_seed(0)

            query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
            key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            attn_mask = None
            dropout_p = 0.0
            is_causal = is_causal

            query_slice = query.chunk(self.world_size, dim=-2)[self.rank]
            key_slice = key.chunk(self.world_size, dim=-2)[self.rank]
            value_slice = value.chunk(self.world_size, dim=-2)[self.rank]

            for _ in range(2 if compile else 1):
                out_slice = func(
                    query_slice,
                    key_slice,
                    value_slice,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

            out_slice_ref = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            ).chunk(self.world_size, dim=-2)[self.rank]

            torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * self.world_size)

    def _test_attn_mode_task(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        attn_mode = self.attn_mode(device)
        if attn_mode is None:
            return

        if is_causal and S_Q != S_KV:
            return

        with torch.no_grad(), torch.cuda.device(self.rank):
            torch.manual_seed(0)

            query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
            key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            attn_mask = None
            dropout_p = 0.0
            is_causal = is_causal

            query_slice = query.chunk(self.world_size, dim=-2)[self.rank]
            key_slice = key.chunk(self.world_size, dim=-2)[self.rank]
            value_slice = value.chunk(self.world_size, dim=-2)[self.rank]

            def func(*args, **kwargs):
                return F.scaled_dot_product_attention(*args, **kwargs)

            if compile:
                func = torch.compile(func, fullgraph=True)

            for _ in range(2 if compile else 1):
                with attn_mode:
                    out_slice = func(
                        query_slice,
                        key_slice,
                        value_slice,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )

            out_slice_ref = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            ).chunk(self.world_size, dim=-2)[self.rank]

            torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * self.world_size)

    def process_task(self, name, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)


class _TestParallelAttn:
    class Runner(ParallelAttnRunner):
        pass

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("compile", [False, True])
    def test_attn_func(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        with self.Runner().start() as runner:
            runner(
                ("_test_attn_func_task", dtype, device, B, H, S_Q, S_KV, D, is_causal, compile),
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("compile", [False, True])
    def test_attn_mode(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        with self.Runner().start() as runner:
            runner(
                ("_test_attn_mode_task", dtype, device, B, H, S_Q, S_KV, D, is_causal, compile),
            )


class TestRingAttn(_TestParallelAttn):
    class Runner(ParallelAttnRunner):
        @property
        def attn_func(self):
            return para_attn_interface.ring_attn_func

        def attn_mode(self, device):
            return para_attn_interface.RingAttnMode()


class TestUlyssesAttn(_TestParallelAttn):
    class Runner(ParallelAttnRunner):
        @property
        def attn_func(self):
            return para_attn_interface.ulysses_attn_func

        def attn_mode(self, device):
            return para_attn_interface.UlyssesAttnMode()


class TestUnifiedAttn(_TestParallelAttn):
    class Runner(ParallelAttnRunner):
        def attn_mode(self, device):
            world_size = self.world_size
            if world_size % 2 == 0:
                mesh_shape = (2, world_size // 2)
            else:
                mesh_shape = (1, world_size)
            mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ring", "ulysses"))
            return para_attn_interface.UnifiedAttnMode(mesh)
