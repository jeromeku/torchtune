from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)
from torch.distributed.tensor import DTensor
from torchao.dtypes.nf4tensor import _INNER_TENSOR_NAMES_FOR_SHARDING, NF4Tensor, to_nf4

from torchtune.modules.low_precision.nf4_linear import FrozenNF4Linear
from torchtune.modules.peft.lora import LoRALinear


@contextmanager
def dist_context():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    yield
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
    with torch.device("cuda"):
        model = LoRALinear(dim, dim, alpha=lora_alpha, rank=lora_rank, quantize_base=True)
        set_requires_grad(model)
    if rank == 0:
        print_nf4_params(model)

    fully_shard(model)

    if rank == 0:
        print("\n ---------- \n", flush=True)
        print_nf4_params(model)
        print("\n ---------- \n", flush=True)


    if rank == 0 and not checkpoint_path.exists():
        print(f"Saving checkpoint to {checkpoint_path.absolute()}", flush=True)

        fs_storage_writer = dcp.FileSystemWriter(checkpoint_path)
        dcp.save(
            state_dict=model.to("cpu").state_dict(),
            storage_writer=fs_storage_writer,
        ) 
    else:
        print(f"Loading checkpoint from {checkpoint_path.absolute()}", flush=True)
        state_dict = model.state_dict()
        fs_storage_reader = dcp.FileSystemReader(checkpoint_path)
        dcp.load(state_dict, storage_reader=fs_storage_reader)

def print_nf4_props(nf4_weight: NF4Tensor):
    NF4_PROPS = ["block_size", "scaler_block_size", "n_blocks"]
    props = [f"{p}: {getattr(nf4_weight, p)}" for p in NF4_PROPS]
    print(f"{', '.join(props)}")
    tensor_props = [(p, getattr(nf4_weight, p)) for p in _INNER_TENSOR_NAMES_FOR_SHARDING]
    for name, tensor in tensor_props:
        print(f"{name}: {tensor.shape} {tensor.dtype} {tensor.device}")

if __name__ == "__main__":

    dim = 512   
    weight = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
    # weight_copy = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
    nf4_weight = to_nf4(weight)
    NF4_PROPS = ["block_size", "scaler_block_size", "n_blocks"]
    print_nf4_props(nf4_weight)

    print("\n ---------- \n", flush=True)
    world_size = 2
    split_size = dim // world_size
    chunks = torch.split(nf4_weight, split_size)
    print_nf4_props(chunks[0])

    print("\n ---------- \n", flush=True)
       #heckpoint_path = Path("checkpoint")


    # with dist_context():
    #     rank = dist.get_rank()
    #     main()    #     main()