import torch
from torch.testing._internal.common_fsdp import FSDPTest


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
            #with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
        except Exception:
            print(f"ERR: subclass doesn't implement {func}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"dispatching {func.name} with args: {[type(arg) for arg in args]} and kwargs: {kwargs}")
        return func(*args, **kwargs)

dtype = torch.bfloat16
device = "cuda"
batch_size = 2
in_features = 256
out_features = 128
inner_tensor = torch.randn(out_features, in_features, dtype=dtype, device=device)
weight = SimpleTensor(inner_tensor)
activation = torch.randn(batch_size, in_features, dtype=dtype, device=device)
out = torch.nn.functional.linear(activation, weight)out = torch.nn.functional.linear(activation, weight)