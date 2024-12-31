import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# Enable memory savings
# pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
video = pipe(
    prompt,
    num_frames=84,
).frames[0]

print("Saving video to mochi.mp4")
export_to_video(video, "mochi.mp4", fps=30)
