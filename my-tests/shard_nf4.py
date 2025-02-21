from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

from torchtune.modules.low_precision.nf4_linear import FrozenNF4Linear
from torchtune.modules.peft.lora import LoRALinear


@contextmanager
def dist_context():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    yield
    all_ranks = dist.get_process_group_ranks(group=dist.group.WORLD)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()

def print_nf4_params(model: nn.Module, print_tensor: bool = False):
    for name, param in model.named_parameters():
        props = f"{name} {type(param)}:\n {param.shape} {param.numel()} {param.dtype} {param.device} {param.requires_grad}"
        if isinstance(param, NF4Tensor):
            props += f"\n {param.quantized_data.shape} {param.quantized_scalers.shape} {param.quantization_factor.shape}"
            print(props)
        elif isinstance(param, DTensor):
            spec = param._spec
            local_tensor = param.to_local()
            props += f"\n {spec}\n {type(local_tensor)} {local_tensor.shape}"
            if isinstance(local_tensor, NF4Tensor):
                props += f"\n {local_tensor.quantized_data.shape} {local_tensor.quantized_scalers.shape} {local_tensor.quantization_factor.shape}"
                print(props)

def set_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(not isinstance(param, NF4Tensor))

def main():
    dim = 512
    lora_rank = 64
    lora_alpha = 2 * lora_rank
    with torch.device("meta"):
        linear = LoRALinear(dim, dim, alpha=lora_alpha, rank=lora_rank, quantize_base=True)
        set_requires_grad(linear)
    if rank == 0:
        print_nf4_params(linear)

    fully_shard(linear)


    if rank == 0:
        print("\n ---------- \n", flush=True)
        print_nf4_params(linear)
        print("\n ---------- \n", flush=True)

if __name__ == "__main__":

    with dist_context():
        rank = dist.get_rank()
        main()
