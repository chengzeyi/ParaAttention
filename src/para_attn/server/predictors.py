import base64
import io
import os
import time

import torch
import torch.distributed as dist
from diffusers import FluxPipeline

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae


class FluxPredictor:
    def setup(self):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        mesh = init_context_parallel_mesh(
            self.pipe.device.type,
            max_ring_dim_size=2,
        )
        parallelize_pipe(
            self.pipe,
            mesh=mesh,
        )
        parallelize_vae(self.pipe.vae, mesh=mesh._flatten())

    def predict(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        height: int,
        width: int,
    ):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil" if dist.get_rank() == 0 else "pt",
            height=height,
            width=width,
        ).images[0]

        if dist.get_rank() == 0:
            byte_array = io.BytesIO()
            image.save(byte_array, format="PNG")
            byte_array.seek(0)
            base64_string = base64.b64encode(byte_array.getvalue()).decode("utf-8")
            return (base64_string, "png")

    def get_test_predict_args(self):
        """
        helpful for unit testing
        """
        return {
            "prompt": "A cat in a hat",
            "negative_prompt": "",
            "guidance_scale": 7,
            "num_inference_steps": 25,
            "height": 512,
            "width": 512,
        }

    def download_model(self):
        """
        helpful for unit testing
        """

        FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")


######################################
# some mock methods for unit testing #
######################################


class MockPipeline:
    def __call__(self, prompt, output_type="pil"):
        self._interrupt = False
        for i in range(10):
            if self._interrupt:
                continue
            time.sleep(1)


class MockPredictor:
    def setup(self):
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        self.pipe = MockPipeline()

    def predict(self, prompt):
        self.pipe(prompt)

        if dist.get_rank() == 0:
            byte_array = io.BytesIO()

            # generate 2^20 random bytes
            byte_array.write(os.urandom(2**20))
            byte_array.seek(0)

            # encode the byte array to a base64 string
            base64_string = base64.b64encode(byte_array.getvalue()).decode("utf-8")

            return (base64_string, "png")

    def get_test_predict_args(self):
        return {"prompt": "A dog in a toupee"}


class MockPredictorSetupError(MockPredictor):
    def setup(self):
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        raise Exception("MockPredictorSetupError")


class MockPredictorPredictError(MockPredictor):
    def predict(self, prompt):
        for i in range(10):
            if i == 5:
                raise Exception("MockPredictorPredictError")
            time.sleep(1)


class LargeOutputPredictor(MockPredictor):
    def setup(self):
        super().setup()

        # set random seed and generate a large output
        torch.manual_seed(42)
        num_bytes = 2**30
        self.large_tensor = torch.randn(num_bytes // 4)  # 1GB large tensor

    def predict(self, prompt):
        # return the large tensor as a base64 string
        byte_array = io.BytesIO()
        byte_array.write(self.large_tensor.numpy().tobytes())
        byte_array.seek(0)
        base64_string = base64.b64encode(byte_array.getvalue()).decode("utf-8")
        return (base64_string, "pt")

    def get_large_tensor(self):
        return self.large_tensor
