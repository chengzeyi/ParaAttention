# Achieving the Fastest HunyuanVideo Inference

## Introduction

During the past year, we have seen the rapid development of video generation models with the release of several open-source models, such as [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo), [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b) and [Mochi](https://huggingface.co/genmo/mochi-1-preview).
It is very exciting to see that open source video models are going to beat closed source.
However, the inference speed of these models is still a bottleneck for real-time applications and deployment.

In this article, we will use [ParaAttention](https://github.com/chengzeyi/ParaAttention), a library implements **Context Parallelism** and **First Block Cache**, as well as other techniques like `torch.compile` and **Dynamic Quantization**, to achieve the fastest inference speed for HunyuanVideo.

## HunyuanVideo Inference with `diffusers`

Like many other generative AI models, HunyuanVideo has its official code repository and is supported by other frameworks like `diffusers` and `ComfyUI`.
In this article, we will focus on optimizing the inference speed of HunyuanVideo with `diffusers`.
To use HunyuanVideo with `diffusers`, we need to install its latest version:

```bash
pip3 install -U diffusers
```

Then, we can load the model and generate video frames with the following code:

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

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
).to("cuda")

pipe.vae.enable_tiling()

begin = time.time()
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
).frames[0]
end = time.time()
print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

However, most people will experience OOM (Out of Memory) errors when running the above code.
This is because the HunyuanVideo transformer model is relatively large and it has a quite large text encoder.
Besides, HunyuanVideo requires a variable length of text conditions and the `diffusers` library implements this feature with a `attn_mask` in `scaled_dot_product_attention`.
The size of `attn_mask` is proportional to the square of the input sequence length, which is crazy when we increase the resolution and the number of frames of the inference!
Luckily, we can use ParaAttention to solve this problem.
In ParaAttention, we patch the original implementation in `diffusers` to cut the text conditions before calling `scaled_dot_product_attention`.
We implement this in our `apply_cache_on_pipe` function so we can call it after loading the model:

```bash
pip3 install -U para-attn
```

```python
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.0)
```

We pass `residual_diff_threshold=0.0` to `apply_cache_on_pipe` to disable the cache mechanism now, because we will enable it later.
Here, we only want it to cut the text conditions to avoid OOM errors.
If you still experience OOM errors, you can try calling `pipe.enable_model_cpu_offload()` after calling `apply_cache_on_pipe`.

This is our base line, on one single NVIDIA H800 GPU, we can generate 129 frames with 720p resolution in 30 inference steps in xx seconds.

## Apply First Block Cache on HunyuanVideo

By caching the output of the transformer blocks in the transformer model and resuing them in the next inference steps, we can reduce the computation cost and make the inference faster.
However, it is hard to decide when to reuse the cache to ensure the quality of the generated video.
Recently, [TeaCache](https://github.com/ali-vilab/TeaCache) suggests that we can use the timestep embedding to approximate the difference among model outputs.
However, TeaCache is still a bit complex as it needs a rescaling strategy to ensure the accuracy of the cache.
In ParaAttention, we find that we can directly use the residual difference of the first transformer block output to approximate the difference among model outputs.
When the difference is small enough, we can reuse the residual difference of previous inference steps, meaning that we in fact skip this denoising step.
This has been proved to be effective in our experiments and we achieve a nearly 2x speedup on HunyuanVideo inference with very good quality.

To apply the first block cache on HunyuanVideo, we can call `apply_cache_on_pipe` with `residual_diff_threshold=0.035`, which is the default value for HunyuanVideo.

```python
apply_cache_on_pipe(pipe, residual_diff_threshold=0.035)
```

Now, on one single NVIDIA H800 GPU, we can generate 129 frames with 720p resolution in 30 inference steps in xx seconds. This is a xx speedup compared to the base line.

## Quantize the model into FP8

To further speed up the inference and reduce memory usage, we can quantize the model into FP8 with dynamic quantization.
If your GPU is not capable of FP8 inference, you can choose to quantize the model int INT8.
[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) provides a really good tutorial on how to quantize models in `diffusers` and achieve a good speedup.
Here, we simply install the latest `torchao` that is capable of quantizing HunyuanVideo:

```bash
pip3 install -U torch torchao
```

We also need to pass the model to `torch.compile` to gain actual speedup.
`torch.compile` with `mode="max-autotune-no-cudagraphs"` or `mode="max-autotune"` can help us to achieve the best performance by generating and selecting the best kernel for the model inference.
The compilation process may take a while, but it is worth it.
In this example, we only quantize the transformer model, but you can also quantize the text encoder to reduce more memory usage.
We also need to notice that the actually compilation process is done on the first time the model is called, so we need to warm up the model to measure the speedup correctly.

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

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
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

from torchao.quantization import autoquant

pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
   pipeline.transformer, mode="max-autotune-no-cudagraphs",
)

pipe.vae.enable_tiling()

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
    ).frames[0]
    end = time.time()
    if i == 0:
        print(f"Warm up time: {end - begin:.2f}s")
    else:
        print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

We can see that dynamic quantization is very effective in speeding up the inference.
We now achieve a xx speedup compared to the base line.

## Parallelize the inference with Context Parallelism

We are not satisfied with the speedup we have achieved so far.
If we want to accelerate the inference further, we can use context parallelism to parallelize the inference.
Luckily, in ParaAttention, we design our API in a compositional way so that we can combine context parallelism with first block cache and dynamic quantization all together.
We provide very detailed instructions and examples of how to scale up the inference with multiple GPUs in our [ParaAttention](https://github.com/chengzeyi/ParaAttention) repository.
Users can easily launch the inference with multiple GPUs by calling `torchrun`.
If there is a need to make the inference process persistent and serviceable, it is suggested to use `torch.multiprocessing` to write your own inference processor, which can eliminate the overhead of launching the process and loading and recompiling the model.

Below is our ultimate code to achieve the fastest HunyuanVideo inference:

```python
import time
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
).to("cuda")

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

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

from torchao.quantization import autoquant

torch._inductor.config.reorder_for_compute_comm_overlap = True

pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
   pipeline.transformer, mode="max-autotune-no-cudagraphs",
)

pipe.vae.enable_tiling()

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"Warm up time: {end - begin:.2f}s")
        else:
            print(f"Time: {end - begin:.2f}s")

if dist.get_rank() == 0:
    print("Saving video to hunyuan_video.mp4")
    export_to_video(output, "hunyuan_video.mp4", fps=15)

dist.destroy_process_group()
```

We save the above code to `run_hunyuan_video.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=8 run_hunyuan_video.py
```

With 8 NVIDIA H800 GPUs, we can generate 129 frames with 720p resolution in 30 inference steps in xx seconds. This is a xx speedup compared to the base line.
