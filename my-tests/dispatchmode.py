import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode_stack,
)
from torchao.dtypes.nf4tensor import (
    _INNER_TENSOR_NAMES_FOR_SHARDING,
    _NF4_QUANT_PROPS,
    NF4Tensor,
    print_tensor_metadata,
    to_nf4,
)


class TorchDispatchNestedMode(TorchDispatchMode):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        modes = _get_current_dispatch_mode_stack()
        for mode in modes:
            print(f"NESTED:mode:{type(mode)} {dir(mode)}")
        return func(*args, **kwargs)


class TorchDispatchLoggingMode(TorchDispatchMode):
    def __torch_dispatch__(cls, func: OpOverload, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        print(
            "-------------------------------- TORCH DISPATCH LOGGING MODE --------------------------------"
        )

        print(f"{func.name}, {types=}, args={[type(arg) for arg in args]}, {kwargs=}")

        for arg in args:
            if isinstance(arg, NF4Tensor):
                print_tensor_metadata(arg)
        return func(*args, **kwargs)


model = torch.nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
dim = 512
x = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
nf4_weight: NF4Tensor = to_nf4(x)

with TorchDispatchLoggingMode():
    chunks = nf4_weight.chunk(2)