import pytest

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

aten = torch.ops.aten
para_attn_ops = torch.ops.para_attn


@pytest.mark.skipif("not torch.cuda.is_available()")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize(
    "backend",
    [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ],
)
@pytest.mark.parametrize(
    "B,H,S_Q,S_KV,D",
    [
        [1, 24, 4096, 4096, 64],
        [1, 24, 4096, 1024, 64],
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_forward_with_lse(dtype, device, backend, B, H, S_Q, S_KV, D, is_causal):
    if is_causal and S_Q != S_KV:
        pytest.skip("is_causal and S_Q != S_KV")

    with torch.no_grad():
        query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
        key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
        value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
        attn_mask = None
        dropout_p = 0.0
        is_causal = is_causal
        with sdpa_kernel([backend]):
            try:
                out = aten.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
            except Exception as e:
                pytest.skip(f"Cannot run {backend} on this configuration: {e}")

        out_new, lse = para_attn_ops.attention_forward_with_lse(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        torch.testing.assert_close(out, out_new)
        assert lse.dtype == torch.float32
        assert lse.shape == (B, H, S_Q)
