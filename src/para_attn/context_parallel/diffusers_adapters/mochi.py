import functools
from typing import List, Optional, Union

import torch
from diffusers import DiffusionPipeline, MochiTransformer3DModel

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: MochiTransformer3DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        **kwargs,
    ):
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)
        encoder_attention_mask = DP.get_assigned_chunk(encoder_attention_mask, dim=0, group=batch_mesh)
        encoder_attention_mask = DP.get_assigned_chunk(encoder_attention_mask, dim=-1, group=seq_mesh)

        with UnifiedAttnMode(mesh):
            output = original_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = DP.get_complete_tensor(sample, dim=-2, group=seq_mesh)
        sample = DP.get_complete_tensor(sample, dim=0, group=batch_mesh)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    transformer.forward = new_forward.__get__(transformer)

    original_time_embed_forward = transformer.time_embed.forward

    @functools.wraps(transformer.time_embed.__class__.forward)
    def new_time_embed_forward(
        self,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        encoder_hidden_states = DP.get_complete_tensor(encoder_hidden_states, dim=-2, group=seq_mesh)
        encoder_attention_mask = DP.get_complete_tensor(encoder_attention_mask, dim=-1, group=seq_mesh)
        with UnifiedAttnMode.disable():
            conditioning, caption_proj = original_time_embed_forward(
                timestep, encoder_hidden_states, encoder_attention_mask, *args, **kwargs
            )
        caption_proj = DP.get_assigned_chunk(caption_proj, dim=-2, group=seq_mesh)
        return conditioning, caption_proj

    transformer.time_embed.forward = new_time_embed_forward.__get__(transformer.time_embed)

    original_rope_forward = transformer.rope.forward

    @functools.wraps(transformer.rope.__class__.forward)
    def new_rope_forward(
        self,
        pos_frequencies: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        *args,
        **kwargs,
    ):
        height *= DP.get_world_size(seq_mesh)
        rope_cos, rope_sin = original_rope_forward(
            pos_frequencies,
            num_frames,
            height,
            width,
            *args,
            **kwargs,
        )
        # dim=1
        n, h, f = rope_cos.shape
        rope_cos = rope_cos.reshape(num_frames, -1, h, f)
        rope_sin = rope_sin.reshape(num_frames, -1, h, f)
        rope_cos = DP.get_assigned_chunk(rope_cos, dim=-3, group=seq_mesh)
        rope_sin = DP.get_assigned_chunk(rope_sin, dim=-3, group=seq_mesh)
        rope_cos = rope_cos.reshape(-1, h, f)
        rope_sin = rope_sin.reshape(-1, h, f)
        return rope_cos, rope_sin

    transformer.rope.forward = new_rope_forward.__get__(transformer.rope)

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
