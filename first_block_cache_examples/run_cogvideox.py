import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    # generator=torch.Generator(device=pipe.device).manual_seed(42),
).frames[0]

print("Saving video to cogvideox.mp4")
export_to_video(video, "cogvideox.mp4", fps=8)
