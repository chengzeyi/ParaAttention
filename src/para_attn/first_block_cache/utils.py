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

    @torch.compiler.disable()
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable()
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


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


def are_two_tensors_similar(output1, output2, *, threshold):
    diff = (output1 - output2).abs().mean().item() / output1.abs().mean().item()
    return diff < threshold


class CachedTransformerBlocks(torch.nn.Module):
    def __init__(self, transformer_blocks, *, transformer=None, residual_diff_threshold):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.residual_diff_threshold = residual_diff_threshold

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            return hidden_states, encoder_hidden_states

        cache_context = get_current_cache_context()
        assert cache_context is not None, "cache_context must be set before"

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = cache_context.get_buffer("first_hidden_states_residual")
        can_use_cache = prev_first_hidden_states_residual is not None and are_two_tensors_similar(
            prev_first_hidden_states_residual, first_hidden_states_residual, threshold=self.residual_diff_threshold
        )

        if self.transformer is not None and getattr(self.transformer, "_is_parallelized", False):
            can_use_cache_t = torch.full((1,), can_use_cache, dtype=torch.bool, device=hidden_states.device)
            can_use_cache_t = DP.get_complete_tensor(can_use_cache_t, dim=0)
            can_use_cache = can_use_cache_t.all().item()

        if can_use_cache:
            hidden_states_residual = cache_context.get_buffer("hidden_states_residual")
            assert hidden_states_residual is not None, "hidden_states_residual must be set before"
            hidden_states = original_hidden_states + hidden_states_residual
        else:
            for block in self.transformer_blocks[1:]:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            hidden_states_residual = hidden_states - original_hidden_states
            cache_context.set_buffer("hidden_states_residual", hidden_states_residual)

        cache_context.set_buffer("first_hidden_states_residual", first_hidden_states_residual)
        return hidden_states, encoder_hidden_states
