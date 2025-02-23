import torch
from torch._ops import OpOverload
from torch.testing._internal.common_fsdp import FSDPTest
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

from torchtune.modules.low_precision import FrozenNF4Linear


class SimpleTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, inner_tensor, *args, **kwargs):
        
        kwargs["device"] = inner_tensor.device
        kwargs["layout"] = inner_tensor.layout
        kwargs["dtype"] = inner_tensor.dtype
        kwargs["requires_grad"] = inner_tensor.requires_grad
        print(f"New SimpleTensor: {kwargs}")
        return torch.Tensor._make_wrapper_subclass(cls, inner_tensor.shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, inner_tensor, *args, **kwargs):
        self.inner_tensor = inner_tensor

    def __repr__(self):
        return f"SimpleTensor({self.inner_tensor.shape})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, simple_tensor, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            print(f"{func.__name__} mat1: {mat1.shape}, simple_tensor: {simple_tensor.shape}")
            return func(mat1, simple_tensor.inner_tensor, bias)
        try:
            print(f"calling {func.__name__} with args: {[type(arg) for arg in args]} and kwargs: {kwargs}")
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except Exception:
            print(f"ERR: subclass doesn't implement {func}")

    def __torch_dispatch__(self, func: OpOverload, types, args=(), kwargs=None):
        
        print(f"dispatching {func._schema.name} {func._opname} {func._overloadname} with {len(args)} args: {[type(arg) for arg in args]} and kwargs: {kwargs}")
        if func is torch.ops.aten.detach.default:
            print(f"returning {args[0]}")
            return args[0]
        return func(*args, **kwargs)

class SimpleLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        self.weight.requires_grad_(False)
        simple_tensor = SimpleTensor(self.weight)
        torch.utils.swap_tensors(self.weight, torch.nn.Parameter(simple_tensor, requires_grad=False))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)

dtype = torch.bfloat16
device = "cuda"
batch_size = 2
in_features = 256
out_features = 128
inner_tensor = torch.randn(out_features, in_features, dtype=dtype, device=device)
model = SimpleLinear(in_features, out_features)
# x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
# out = model(x)
# print(out)

