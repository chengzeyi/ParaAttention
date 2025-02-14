import torch
from packaging import version

from torch.overrides import TorchFunctionMode

import para_attn


@torch.compiler.assume_constant_result
def _torch_version_check(op, v):
    return getattr(version.parse(torch.__version__), op)(version.parse(v))


@torch.compiler.assume_constant_result
def _get_force_dispatch_to_custom_ops():
    return para_attn.config.attention.force_dispatch_to_custom_ops


class BaseTorchFunctionMode(TorchFunctionMode):
    @torch.compiler.disable
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.Tensor.unflatten:
            # Reason: Unsupported: non-function or method super: <method 'unflatten' of 'torch._C.TensorBase' objects>
            return torch.unflatten(*args, **kwargs)

        return func(*args, **kwargs)
