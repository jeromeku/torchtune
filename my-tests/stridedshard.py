import itertools
import time
import warnings
from contextlib import contextmanager

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor._utils import (  # compute_local_shape,
    compute_local_shape_and_global_offset,
)

#from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import Shard, _StridedShard
from torch.testing._internal.common_utils import run_tests

#from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

warnings.filterwarnings("ignore", category=UserWarning)

def dist_print(*msg, delay=1, rank0_only=False):
    rank = torch.distributed.get_rank()
    if rank0_only and rank != 0:
        return
    time.sleep(delay * rank)

    print(f"[rank{rank}]: ", *msg, flush=True)


@contextmanager
def dist_context():
    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    dist_print(f"rank {rank} initialized with world size {torch.distributed.get_world_size()}")
    yield
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_fsdp_tp_meta_compute():
    # FSDP + TP sharding
    world_size = torch.distributed.get_world_size()
    tp_size = 4
    dp_size = world_size // tp_size
    global_mesh = init_device_mesh(
        "cpu", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    
    t = torch.arange(world_size * 2).reshape(world_size * 2, 1)
    # local shard shape is [2, 2]
    global_tensor_shape = torch.Size([2 * world_size, 1])
    placements = [_StridedShard(0, split_factor=tp_size), Shard(0)]

    local_shape, global_offset = compute_local_shape_and_global_offset(
        global_tensor_shape, global_mesh, placements
    )
    assert global_mesh.get_coordinate is not None
    dp_rank = global_mesh.get_local_rank("dp")
    tp_rank = global_mesh.get_local_rank("tp")
    shard_idx_on_dim_0 = tp_rank * dp_size + dp_rank
   # dist_print(f"stridedshard: {dp_rank=}, {tp_rank=}, {shard_idx_on_dim_0=} {global_offset=}")

    distributed_tensor = distribute_tensor(t, global_mesh, placements)
#    dist_print(f"distributed_tensor.shape: {distributed_tensor.shape}", rank0_only=True)

    local_tensor = distributed_tensor.to_local()
    dist_print(f"local: {local_tensor.view(-1).tolist()}")

    torch.distributed.barrier()
    tp_mesh = global_mesh["tp"]
    dp_mesh = global_mesh["dp"]
    dist_print(f"t: {t.view(-1).tolist()}", rank0_only=True)
    tp_shard = distribute_tensor(t, tp_mesh, [Shard(0)])
    tp_shard = tp_shard.to_local()
    dist_print(f"tp_shard: {tp_shard.view(-1).tolist()}")
    dp_shard = distribute_tensor(tp_shard, dp_mesh, [Shard(0)])
    dp_shard = dp_shard.to_local()
    dist_print(f"local: {local_tensor.view(-1).tolist()} dp_shard: {dp_shard.view(-1).tolist()}")

if __name__ == "__main__":
    with dist_context():
        test_fsdp_tp_meta_compute()