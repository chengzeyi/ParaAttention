import functools
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline, MochiTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging, scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def parallelize_transformer(transformer: MochiTransformer3DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        temb, encoder_hidden_states = self.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        temb = DP.get_assigned_chunk(temb, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = DP.get_assigned_chunk(freqs, dim=0, group=seq_mesh)
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        with UnifiedAttnMode(mesh):
            hidden_states, encoder_hidden_states = self.call_transformer_blocks(
                hidden_states, encoder_hidden_states, temb, encoder_attention_mask, image_rotary_emb
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    transformer.forward = new_forward.__get__(transformer)

    def call_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )

        return hidden_states, encoder_hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    original_call = pipe.__class__.__call__

    if not getattr(original_call, "_is_parallelized", False):

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, **kwargs)

    pipe._is_parallelized = True

    return pipe
