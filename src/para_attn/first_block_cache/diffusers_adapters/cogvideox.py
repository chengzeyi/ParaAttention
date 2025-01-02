import functools
import unittest

import torch
from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline

from para_attn.first_block_cache import utils


def apply_cache_on_transformer(
    transformer: CogVideoXTransformer3DModel,
    *,
    residual_diff_threshold=0.025,
):
    cached_transformer_blocks = torch.nn.ModuleList(
        [
            utils.CachedTransformerBlocks(
                transformer.transformer_blocks,
                transformer=transformer,
                residual_diff_threshold=residual_diff_threshold,
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    return transformer


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    **kwargs,
):
    original_call = pipe.__class__.__call__

    if not getattr(original_call, "_is_cached", False):

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with utils.cache_context(utils.create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call

        new_call._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer, **kwargs)

    pipe._is_cached = True

    return pipe
