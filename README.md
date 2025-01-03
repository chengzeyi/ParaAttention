# ParaAttention

Context parallel attention that accelerates DiT model inference with dynamic caching,
supporting both [**Ulysses Style**](https://arxiv.org/abs/2309.14509) and [**Ring Style**](https://arxiv.org/abs/2310.01889) parallelism.

🔥[Fastest HunyuanVideo Inference with Context Parallelism and First Block Cache on NVIDIA L20 GPUs](doc/fastest_hunyuan_video.md)🔥

This aims to provide:

- [x] An easy to use interface to speed up model inference with context parallel, dynamic caching and `torch.compile`. Make **`FLUX`**, **`HunyuanVideo`** and **`Mochi`** inference much faster losslessly.
- [x] A unified interface to run context parallel attention (***cfg-ulysses-ring***), as well as keeping the maximum performance while working with `torch.compile`
- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.

What's different from other implementations:

- No unnecessary graph breaks during `torch.compile`. All the heavy computations are captured in a single graph and get the maximum opportunity to be optimized. This makes it possible for the backend compiler to optimize the graph more effectively, for example, by overlapping the computation and communication.
- Easy to use. You don't need to change the code of the model to enable context parallelism. Instead, you only need to call a function to parallelize the model.
- Easy to use, too. If you want to use context parallelism with your custom model, you only need to wrap the call with our special `TorchFunctionMode` context manager.
- Easy to adjust. You can adjust the parallelism style and the mesh shape with a few lines of code.

# Key Features

### Context Parallelism

**Context Parallelism** (CP) is a method for parallelizing the processing of neural network activations across multiple GPUs by partitioning the input tensors along the sequence dimension.
Unlike Sequence Parallelism (SP) that partitions the activations of specific layers, CP divides the activations of all layers.
In `ParaAttention`, we are able to parallelize the attention layer with a mixture of Ulysses Style and Ring Style parallelism, called Unified Attention.
This allows us to achieve the best performance with different models and different hardware configurations.
We also provide a unified interface to parallelize the model inference.

### First Block Cache (Our Dynamic Caching)

Inspired by [TeaCache](https://github.com/ali-vilab/TeaCache) and other denoising caching algorithms, we introduce **First Block Cache** (FBC) to use the residual output of the first transformer block as the cache indicator.
If the difference between the current and the previous residual output of the first transformer block is small enough, we can reuse the previous final residual output and skip the computation of all the following transformer blocks.
This can significantly reduce the computation cost of the model, achieving a speedup of up to 2x while maintaining high accuracy.

| HunyuanVideo without FBC | HunyuanVideo with FBC |
| - | - |
| [Original](https://github.com/user-attachments/assets/883d771a-e74e-4081-aa2a-416985d6c713) | [FBC](https://github.com/user-attachments/assets/f77c2f58-2b59-4dd1-a06a-a36974cb1e40) |

# Officially Supported Models

## Context Parallelism with First Block Cache

You could run the following examples with `torchrun` to enable context parallelism with dynamic caching.
You can modify the code to enable `torch.compile` to further accelerate the model inference.
If you want quantization, please refer to [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) for more information.
For example, to run FLUX with 2 GPUs:

**NOTE**: To measure the performance correctly with `torch.compile`, you need to warm up the model by running it for a few iterations before measuring the performance.

```bash
# Use --nproc_per_node to specify the number of GPUs
torchrun --nproc_per_node=2 parallel_examples/run_flux.py
```

- [FLUX](parallel_examples/run_flux.py)
- [HunyuanVideo🚀](parallel_examples/run_hunyuan_video.py)
- [Mochi](parallel_examples/run_mochi.py)
- [CogVideoX](parallel_examples/run_cogvideox.py)

## Single GPU Inference with First Block Cache

You can also run the following examples with a single GPU and enable the First Block Cache to speed up the model inference.

```bash
python3 first_block_cache_examples/run_hunyuan_video.py
```

- [HunyuanVideo🚀](first_block_cache_examples/run_hunyuan_video.py)
- [Mochi](first_block_cache_examples/run_mochi.py)
- [CogVideoX](first_block_cache_examples/run_cogvideox.py)

**NOTE**: To run `HunyuanVideo`, you need to install `diffusers` from its latest master branch.
It is suggested to run `HunyuanVideo` with GPUs with at least 48GB memory, or you might experience OOM errors,
and the performance might be worse due to frequent memory re-allocation.

# Performance

## Context Parallelism (without First Block Cache)

| Model | GPU | Method | Wall Time (s) | Speedup |
| --- | --- | --- | --- | --- |
| FLUX.1-dev | A100-SXM4-80GB | Baseline | 13.843 | 1.00x |
| FLUX.1-dev | A100-SXM4-80GB | `torch.compile` | 9.997 | 1.38x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ring)` | 8.307 | 1.66x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ring)` + `torch.compile` | 5.775 | 2.39x |
| FLUX.1-dev | A100-SXM4-80GB x 4 | `para-attn (ulysses + ring)` | 6.157 | 2.25x |
| FLUX.1-dev | A100-SXM4-80GB x 4 | `para-attn (ulysses + ring)` + `torch.compile` | 3.557 | 3.89x |
| mochi-1-preview | A100-SXM4-80GB | Baseline | 196.534 | 1.00x |
| mochi-1-preview | A100-SXM4-80GB | `torch.compile` | 149.868 | 1.31x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (cfg)` | 105.438 | 1.86x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ulysses)` | 110.146 | 1.78x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ring)` | 109.435 | 1.80x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (cfg)` + `torch.compile` | 81.913 | 2.40x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ulysses)` + `torch.compile` | 83.912 | 2.34x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ring)` + `torch.compile` | 82.176 | 2.39x |
| mochi-1-preview | A100-SXM4-80GB x 4 | `para-attn (cfg + ring)` | 61.206 | 3.21x |
| mochi-1-preview | A100-SXM4-80GB x 4 | `para-attn (cfg + ring)` + `torch.compile` | 47.100 | 4.17x |

**NOTE**: The speedup of iterations per second is generally higher than the speedup of wall time, because the wall time includes the overhead of calling the text encoder and vae decoder.

# Installation

## Install from PyPI

```bash
pip3 install 'torch==2.5.0'
pip3 install para-attn
```

## Local Installation

```bash
git clone https://github.com/chengzeyi/ParaAttention.git
cd ParaAttention
git submodule update --init --recursive

pip3 install 'torch==2.5.0'
pip3 install 'setuptools>=64' 'setuptools_scm>=8'

# Pass --no-use-pep517 to speed up rebuild by using the legacy build system
# which doesn't use a one-time tmp directory for the build
pip3 install -e '.[dev]' --no-build-isolation
# Or:
# python3 setup.py develop

# Code formatting and linting
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

# Usage

## All Examples

Please refer to examples in the `parallel_examples` and `first_block_cache_examples` directories.

### Parallelize Models

| Model | Command |
| - | - |
| `FLUX` | `torchrun --nproc_per_node=2 parallel_examples/run_flux.py` |
| `HunyuanVideo` | `torchrun --nproc_per_node=2 parallel_examples/run_hunyuan_video.py` |
| `Mochi` | `torchrun --nproc_per_node=2 parallel_examples/run_mochi.py` |
| `CogVideoX` | `torchrun --nproc_per_node=2 parallel_examples/run_cogvideox.py` |

### Apply First Block Cache

| Model | Command |
| - | - |
| `FLUX` | `python3 first_block_cache_examples/run_flux.py` |
| `HunyuanVideo` | `python3 first_block_cache_examples/run_hunyuan_video.py` |
| `Mochi` | `python3 first_block_cache_examples/run_mochi.py` |
| `CogVideoX` | `python3 first_block_cache_examples/run_cogvideox.py` |

## Parallelize VAE

VAE can be parallelized with `para_attn.parallel_vae.diffusers_adapters.parallelize_vae`.
Currently, only `AutoencoderKL` and `AutoencoderKLHunyuanVideo` are supported.

``` python
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

parallelize_vae(vae)
```

## Run Unified Attention (Hybird Ulysses Style and Ring Style) with `torch.compile`

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface

dist.init_process_group()
world_size = dist.get_world_size()
rank = dist.get_rank()

assert world_size <= torch.cuda.device_count()
if world_size % 2 == 0:
    mesh_shape = (2, world_size // 2)
else:
    mesh_shape = (1, world_size)

B, H, S_Q, S_KV, D = 2, 24, 4096, 4096, 64
dtype = torch.float16
device = "cuda"

def func(*args, **kwargs):
    return F.scaled_dot_product_attention(*args, **kwargs)

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# func = torch.compile(func)

with torch.no_grad(), torch.cuda.device(rank):
    torch.manual_seed(0)

    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    attn_mask = None
    dropout_p = 0.0
    is_causal = False

    query_slice = query.chunk(world_size, dim=-2)[rank]
    key_slice = key.chunk(world_size, dim=-2)[rank]
    value_slice = value.chunk(world_size, dim=-2)[rank]

    for _ in range(2):
        mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ring", "ulysses"))
        with para_attn_interface.UnifiedAttnMode(mesh):
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
    ).chunk(world_size, dim=-2)[rank]

    torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * world_size)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```

# Run Tests

```bash
DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=3000 pytest tests
```
