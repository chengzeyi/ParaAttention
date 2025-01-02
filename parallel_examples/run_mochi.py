import torch
import torch.distributed as dist
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
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
