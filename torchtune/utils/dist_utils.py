import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


def dist_print(*msg, rank0_only=True, delay=0):
    rank = dist.get_rank()
    if rank0_only and rank != 0:
        return

    time.sleep(delay * rank)
    print(f"[rank{rank}]: ", *msg, flush=True)

def dist_breakpoint(rank=0):
    if dist.get_rank() == rank:
        breakpoint()
    dist.barrier()

def print_fsdp_param_group(model: FSDPModule = None, pg: FSDPParamGroup = None):
    assert (model is not None) or (pg is not None) 
    if model is not None:
        pg = model._get_fsdp_state()._fsdp_param_group

    props_to_print = [
        "is_sharded",
        "modules",
        "fsdp_params",
        "_sharded_state",
        "comm_ctx",
    ]
    props = "\n".join([f"{prop}: {getattr(pg, prop)}" for prop in props_to_print])
    dist_print(f"{pg}:\n{props}")
    for param in pg.fsdp_params:
        # if param.sharded_state == ShardedState.SHARDED:
        dist_print(
            f"{param}: {param.sharded_state=} {param.sharded_size=}, {param._orig_size=}"
        )
        # pg.unshard()
        # dist_print(f"After unsharding {param}: {param._unsharded_param=} {param.sharded_state=}")

@contextmanager
def dist_context():
    rank_ = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", None))
    print(f"Detected {rank_} {world_size=}")

    dist.init_process_group("nccl", device_id=torch.device(rank_))
    rank = int(dist.get_rank())
    dist_print(f"Rank {rank} init...", rank0_only=False, delay=0.1)
    torch.cuda.set_device(rank)
    dist.barrier()
    yield
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with dist_context():
        dist_breakpoint()

