import time

import torch
import torch.distributed as dist


def dist_print(*msgs, delay=.5, rank0_only=False):
    rank = dist.get_rank()
    if rank0_only and rank != 0:
        return

    time.sleep(delay * rank)
    prefix= f"[rank{rank}]: "
    print(prefix, *msgs, flush=True)

def dist_breakpoint(delay=10):
    rank = dist.get_rank()
    if rank == 0:
        breakpoint()
    torch.distributed.barrier()