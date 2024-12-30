# ParaAttention

Context parallel attention that accelerates DiT model inference,
supporting both [**Ulysses Style**](https://arxiv.org/abs/2309.14509) and [**Ring Style**](https://arxiv.org/abs/2310.01889) parallelism.

This aims to provide:

- [x] An easy to use interface to speed up model inference with context parallel and `torch.compile`. Make **`FLUX`**, **`HunyuanVideo`** and **`Mochi`** inference much faster losslessly.
- [x] A unified interface to run context parallel attention (***cfg-ulysses-ring***), as well as keeping the maximum performance while working with `torch.compile`
- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.

What's different from other implementations:

- No unnecessary graph breaks during `torch.compile`. All the heavy computations are captured in a single graph and get the maximum opportunity to be optimized. This makes it possible for the backend compiler to optimize the graph more effectively, for example, by overlapping the computation and communication.
- Easy to use. You don't need to change the code of the model to enable context parallelism. Instead, you only need to call a function to parallelize the model.
- Easy to use, too. If you want to use context parallelism with your custom model, you only need to wrap the call with our special `TorchFunctionMode` context manager.
- Easy to adjust. You can adjust the parallelism style and the mesh shape with a few lines of code.

# Officially Supported Models

You could run the following examples with `torchrun`.
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

**NOTE**: To run `HunyuanVideo`, you need to install `diffusers` from its latest master branch.
It is suggested to run `HunyuanVideo` with GPUs with at least 48GB memory, or you might experience OOM errors,
and the performance might be worse due to frequent memory re-allocation.

# Performance

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

## Run FLUX.1-dev with Parallel Inference

```python
import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).images[0]

if dist.get_rank() == 0:
    print("Saving image to flux.png")
    image.save("flux.png")

dist.destroy_process_group()
```

Save the above code to `run_flux.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 run_flux.py
```

## Run HunyuanVideo🚀 with Parallel Inference

**NOTE**: To run `HunyuanVideo`, you need to install `diffusers` from its latest master branch.
It is suggested to run `HunyuanVideo` with GPUs with at least 48GB memory, or you might experience OOM errors,
and the performance might be worse due to frequent memory re-allocation.

```python
import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

dist.init_process_group()

# [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
torch.backends.cuda.enable_cudnn_sdp(False)

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
pipe.vae.enable_tiling(
    # Make it runnable on GPUs with 48GB memory
    # tile_sample_min_height=128,
    # tile_sample_stride_height=96,
    # tile_sample_min_width=128,
    # tile_sample_stride_width=96,
    # tile_sample_min_num_frames=32,
    # tile_sample_stride_num_frames=24,
)

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to hunyuan_video.mp4")
    export_to_video(output, "hunyuan_video.mp4", fps=15)

dist.destroy_process_group()
```

Save the above code to `run_hunyuan_video.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 run_hunyuan_video.py
```

## Run Mochi with Parallel Inference

```python
import torch
import torch.distributed as dist
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

dist.init_process_group()

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.bfloat16,
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_batch_dim_size=2,
        max_ring_dim_size=2,
    ),
)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
pipe.enable_vae_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
video = pipe(
    prompt,
    num_frames=84,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to mochi.mp4")
    export_to_video(video, "mochi.mp4", fps=30)

dist.destroy_process_group()
```

Save the above code to `run_mochi.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 run_mochi.py
```

## Parallelize VAE

VAE can be parallelized with `para_attn.parallel_vae.diffusers_adapters.parallelize_vae`.
Currently, only `AutoencoderKL` is supported.

``` python
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL

dist.init_process_group()

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to(f"cuda:{dist.get_rank()}")

from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

parallelize_vae(vae)
```

## All Examples

| Model | Command |
| - | - |
| `FLUX` | `torchrun --nproc_per_node=2 parallel_examples/run_flux.py` |
| `HunyuanVideo` | `torchrun --nproc_per_node=2 parallel_examples/run_hunyuan_video.py` |
| `Mochi` | `torchrun --nproc_per_node=2 parallel_examples/run_mochi.py` |
| `CogVideoX` | `torchrun --nproc_per_node=2 parallel_examples/run_cogvideox.py` |

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
