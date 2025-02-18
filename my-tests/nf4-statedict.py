import torch
import torch.nn as nn
from torchao.dtypes.nf4tensor import _INNER_TENSOR_NAMES_FOR_SHARDING, NF4Tensor, to_nf4

DEVICE = "cuda"

class NF4Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nf4_weight = nn.Parameter(to_nf4(self.weight), requires_grad=False)
        torch.utils.swap_tensors(self.weight, nf4_weight)

def test_nf4_statedict(dim=128, dtype=torch.bfloat16):
    model = NF4Linear(dim, dim, dtype=dtype, bias=False).to(DEVICE)
    print(model.weight)
    
    for name in _INNER_TENSOR_NAMES_FOR_SHARDING:
        print(name, getattr(model.weight, name).shape)

    x = torch.randn(dim, dim, dtype=dtype, device=DEVICE)
    out = model(x)
    print(out.dtype)
    sd = model.state_dict()
    for k,v in sd.items():
        print(k, type(v))
    new_model = NF4Linear(dim, dim, dtype=dtype, bias=False).to(DEVICE)
    breakpoint()
    new_model.load_state_dict(sd)
    print(new_model.weight)
    original_nf4: NF4Tensor = sd["weight"]
    original_wt = original_nf4.get_original_weight()
    new_nf4 = new_model.weight
    new_wt = new_nf4.get_original_weight()
    assert torch.allclose(original_wt, new_wt)
    print("allclose")
if __name__ == "__main__":
    test_nf4_statedict()
