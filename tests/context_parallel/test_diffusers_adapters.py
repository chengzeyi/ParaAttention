import contextlib

import time

import pytest
import pytest_html

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms


class DiffusionPipelineTest(DTensorTestBase):
    @property
    def world_size(self):
        device_count = torch.cuda.device_count()
        return device_count

    def mesh(self, device, use_batch, use_ring):
        from para_attn.context_parallel import init_context_parallel_mesh

        max_batch_dim_size = None
        if use_batch:
            max_batch_dim_size = 2
        max_ring_dim_size = None
        if use_ring:
            max_ring_dim_size = 2
        mesh = init_context_parallel_mesh(
            device, max_batch_dim_size=max_batch_dim_size, max_ring_dim_size=max_ring_dim_size
        )
        return mesh

    def new_pipe(self, dtype, device, rank):
        raise NotImplementedError

    def call_pipe(self, pipe, **kwargs):
        raise NotImplementedError

    @property
    def enable_vae_parallel(self):
        return False

    def _test_benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        with torch.cuda.device(self.rank) if device == "cuda" else contextlib.nullcontext():
            torch.manual_seed(0)

            pipe = self.new_pipe(dtype, device)

            if parallelize:
                from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

                mesh = self.mesh(device, use_batch=use_batch, use_ring=use_ring)
                parallelize_pipe(pipe, mesh=mesh)

                if self.enable_vae_parallel:
                    from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

                    parallelize_vae(pipe.vae, mesh=mesh._flatten())

            if compile:
                if parallelize:
                    torch._inductor.config.reorder_for_compute_comm_overlap = True
                # If cudagraphs is enabled and parallelize is True, the test will hang indefinitely
                # after the last iteration.
                pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

            for i in range(2):
                call_kwargs = {}
                if i == 0:
                    call_kwargs["num_inference_steps"] = 1
                begin = time.time()
                output = self.call_pipe(pipe, **call_kwargs)
                end = time.time()
                msg = f"{'Warm-up' if i == 0 else 'Inference'} time taken: {end - begin:.3f} seconds"
                if self.rank == 0:
                    print(msg)
                    if i != 0:
                        if hasattr(output, "images"):
                            image = output.images[0]
                        elif hasattr(output, "frames"):
                            video = output.frames[0]
                            image = video[0]

class FluxPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        return pipe

    def call_pipe(self, pipe, **kwargs):
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 28
        return pipe(
            "A cat holding a sign that says hello world",
            output_type="pil" if self.rank == 0 else "pt",
            **kwargs,
        )

    @property
    def enable_vae_parallel(self):
        return True

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_batch,use_ring",
        [
            [False, False, False, False],
            [False, True, False, False],
            [True, False, False, True],
            [True, True, False, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_batch, use_ring)


class MochiPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import MochiPipeline

        pipe = MochiPipeline.from_pretrained(
            "genmo/mochi-1-preview",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        pipe.enable_vae_tiling()
        return pipe

    def call_pipe(self, pipe, **kwargs):
        return pipe(
            "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
            num_frames=84,
            output_type="pil" if self.rank == 0 else "pt",
            **kwargs,
        )

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_batch,use_ring",
        [
            [False, False, False, False],
            [False, True, False, False],
            [True, False, True, True],
            [True, True, True, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_batch, use_ring)


class CogVideoXPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import CogVideoXPipeline

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        return pipe

    def call_pipe(self, pipe, **kwargs):
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 50
        prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
        return pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_frames=49,
            guidance_scale=6,
            output_type="pil" if self.rank == 0 else "pt",
            **kwargs,
        )

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_batch,use_ring",
        [
            [False, False, False, False],
            [False, True, False, False],
            [True, False, True, True],
            [True, True, True, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_batch, use_ring)


class HunyuanVideoPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

        # [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
        # torch.backends.cuda.enable_cudnn_sdp(False)

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
            torch_dtype=dtype,
            revision="refs/pr/18",
        ).to(f"{device}:{self.rank}")

        pipe.vae.enable_tiling(
            # Make it runnable on GPUs with 48GB memory
            # tile_sample_min_height=128,
            # tile_sample_stride_height=96,
            # tile_sample_min_width=128,
            # tile_sample_stride_width=96,
            # tile_sample_min_num_frames=32,
            # tile_sample_stride_num_frames=24,
        )

        # Fix OOM because of awful inductor lowering of attn_bias of _scaled_dot_product_efficient_attention
        # import para_attn
        #
        # para_attn.config.attention.force_dispatch_to_custom_ops = True

        return pipe

    def call_pipe(self, pipe, **kwargs):
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 30
        return pipe(
            prompt="A cat walks on the grass, realistic",
            height=720,
            width=1280,
            num_frames=129,
            output_type="pil" if self.rank == 0 else "pt",
            **kwargs,
        )

    @property
    def enable_vae_parallel(self):
        return True

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_batch,use_ring",
        [
            [False, False, False, False],
            [False, True, False, False],
            [True, False, False, False],
            [True, True, False, False],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_batch, use_ring)


instantiate_parametrized_tests(DiffusionPipelineTest)
instantiate_parametrized_tests(FluxPipelineTest)
instantiate_parametrized_tests(MochiPipelineTest)
instantiate_parametrized_tests(CogVideoXPipelineTest)
instantiate_parametrized_tests(HunyuanVideoPipelineTest)

if __name__ == "__main__":
    run_tests()
