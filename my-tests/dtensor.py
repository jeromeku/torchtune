import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch._dynamo.testing import CompileCounterWithBackend
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard


def dist_print(*msg, delay=.5, rank0_only=False):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank0_only and rank != 0:
        return
    time.sleep(delay * rank)
    print(f"[rank{rank}]", *msg, flush=True)

@contextmanager
def dist_context():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    dist_print(f"rank {rank} of {world_size} initialized")
    yield
    dist.barrier()
    dist.destroy_process_group()

def print_tensor(name, tensor, rank0_only=False):
    dist_print(f"{name}: {type(tensor)} {tensor.shape} {tensor.dtype} {tensor.device}", rank0_only=rank0_only)

def main():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    def fn(x):
        dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        dt = dt + 1
        dt = dt.redistribute(mesh, [Replicate()])
        dt = dt.to_local()
        return dt

    x = torch.ones(4, 8, device="cuda")
    ref = fn(x)
    print_tensor("ref", ref, rank0_only=True)
    cnt = CompileCounterWithBackend("aot_eager")
    opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
    res = opt_fn(x)
    # print_tensor("res", res, rank0_only=True)

if __name__ == "__main__":
    with dist_context():
        main()
