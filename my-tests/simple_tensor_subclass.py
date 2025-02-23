from io import BytesIO

import torch
from torch._ops import OpOverload
from torch.testing._internal.common_fsdp import FSDPTest
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

from torchtune.modules.low_precision import FrozenNF4Linear

aten = torch.ops.aten


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

    def __tensor_flatten__(self):
        return ["inner_tensor"], None
    
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return SimpleTensor(inner_tensors["inner_tensor"])
    
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
        except Exception as e:
            print(f"ERR: subclass doesn't implement {func}")
            raise e

    def __torch_dispatch__(self, func: OpOverload, types, args=(), kwargs=None):
        
        FUNCS = [aten.detach.default, aten.copy_.default]
        print(f"dispatching {func._schema.name} {func._opname} {func._overloadname} with {len(args)} args: {[type(arg) for arg in args]} and kwargs: {kwargs}")
        print(f"Func in impelmented funcs: {func in FUNCS}")
        if func is torch.ops.aten.detach.default:
            print(f"returning {args[0]}")
            return args[0]
        if func is aten.copy_.default:
            print(f"copying {args[0]} to {args[1]}")
            original = args[0]
            copy_in = args[1]
            original.inner_tensor.copy_(copy_in.inner_tensor)
            return
        return func(*args, **kwargs)
torch.serialization.add_safe_globals([SimpleTensor])

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

print("\n=================== Creating model =================================\n")
model = SimpleLinear(in_features, out_features)
print(f"Inner tensor: {model.weight.inner_tensor.shape}")

print("\n=================== Saving state dict =================================\n")
state_dict = model.state_dict()
for k,v in state_dict.items():
    print(f"{k}: {type(v)}")
print(f"Saved state dict inner tensor shape: {state_dict['weight'].inner_tensor.shape}")

# buffer = BytesIO()
# torch.save(state_dict, buffer)
# buffer.seek(0)
# print(f" =============== Loading state dict =================")
# state_dict = torch.load(buffer, weights_only=False)
# print(f" =============== Loaded state dict =================")
# for k,v in state_dict.items():
#     print(f"{k}: {type(v)}")
# breakpoint()
# print("\n=================== Loading state dict =================================\n")
# model.load_state_dict(state_dict, assign=True)

# # x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
# # out = model(x)
# # print(out)
