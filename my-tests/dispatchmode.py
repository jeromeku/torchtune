import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode_stack,
)


class TorchDispatchNestedMode(TorchDispatchMode):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        modes = _get_current_dispatch_mode_stack()
        for mode in modes:
            print(f"NESTED:mode:{type(mode)} {dir(mode)}")
        return func(*args, **kwargs)


class TorchDispatchLoggingMode(TorchDispatchMode):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        arg_shape = (
            args[0].shape
            if len(args) > 0 and isinstance(args[0], torch.Tensor)
            else None
        )
        print(f"current dispatch mode stack: {_get_current_dispatch_mode_stack()}")
        print(
            f"ATEN_FUNC {func=}, {types=}, {[type(arg) for arg in args]}, {kwargs=}, args[0] shape: {arg_shape}"
            )

        return func(*args, **kwargs)

model = torch.nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
x = torch.randn(10, 64, device="cuda", dtype=torch.bfloat16)

with TorchDispatchLoggingMode():
    with TorchDispatchNestedMode():
        model(x)
    
