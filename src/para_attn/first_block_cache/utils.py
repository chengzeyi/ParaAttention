import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict

import torch

import para_attn.primitives as DP


@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    return CacheContext()


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    if parallelized:
        mean_diff = DP.all_reduce_sync(mean_diff, "avg")
        mean_t1 = DP.all_reduce_sync(mean_t1, "avg")
    diff = mean_diff / mean_t1
    return diff.item() < threshold


@torch.compiler.disable
def apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states):
    hidden_states_residual = get_buffer("hidden_states_residual")
    assert hidden_states_residual is not None, "hidden_states_residual must be set before"
    hidden_states = hidden_states_residual + hidden_states

    encoder_hidden_states_residual = get_buffer("encoder_hidden_states_residual")
    assert encoder_hidden_states_residual is not None, "encoder_hidden_states_residual must be set before"
    encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable
def get_can_use_cache(first_hidden_states_residual, threshold, parallelized=False):
    prev_first_hidden_states_residual = get_buffer("first_hidden_states_residual")
    can_use_cache = prev_first_hidden_states_residual is not None and are_two_tensors_similar(
        prev_first_hidden_states_residual,
        first_hidden_states_residual,
        threshold=threshold,
        parallelized=parallelized,
    )
    return can_use_cache


class CachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        residual_diff_threshold,
        return_hidden_states_first=True,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.return_hidden_states_first = return_hidden_states_first

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
            return (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
        if not self.return_hidden_states_first:
            hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=self.transformer is not None and getattr(self.transformer, "_is_parallelized", False),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                hidden_states, encoder_hidden_states
            )
        else:
            set_buffer("first_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_remaining_transformer_blocks(hidden_states, encoder_hidden_states, *args, **kwargs)
            set_buffer("hidden_states_residual", hidden_states_residual)
            set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        return (
            (hidden_states, encoder_hidden_states)
            if self.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )

    def call_remaining_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self.transformer_blocks[1:]:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )

        # hidden_states_shape = hidden_states.shape
        # encoder_hidden_states_shape = encoder_hidden_states.shape
        hidden_states = hidden_states.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        # hidden_states = hidden_states.contiguous()
        # encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states

        hidden_states_residual = hidden_states_residual.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states_residual = (
            encoder_hidden_states_residual.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual
