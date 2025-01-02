import torch
import torch.distributed as dist
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
).to("cuda")

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

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    # generator=torch.Generator(device=pipe.device).manual_seed(42),
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to cogvideox.mp4")
    export_to_video(video, "cogvideox.mp4", fps=8)

dist.destroy_process_group()
