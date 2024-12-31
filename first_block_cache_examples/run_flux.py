import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# pipe.enable_model_cpu_offload()

# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
).images[0]

print("Saving image to flux.png")
image.save("flux.png")
