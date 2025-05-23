import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist


def print_dist(*msg, rank0_only=True, delay=0):
    rank = dist.get_rank()
    if rank0_only and rank != 0:
        return

    time.sleep(delay * rank)
    print(*msg, flush=True)

def dist_breakpoint(rank=0):
    if dist.get_rank() == rank:
        breakpoint()
    dist.barrier()
    
@contextmanager
def dist_context():
    rank_ = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", None))
    print(f"Detected {rank_} {world_size=}")

    dist.init_process_group("nccl", device_id=torch.device(rank_))
    rank = dist.get_rank()
    print_dist(f"Rank {rank} init...", rank0_only=False, delay=0.1)
    dist.barrier()
    yield
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with dist_context():
        dist_breakpoint()

