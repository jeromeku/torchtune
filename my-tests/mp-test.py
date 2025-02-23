import os
import time
from contextlib import contextmanager
from functools import partial

import torch
import torch.multiprocessing as mp

filename = "mp_store"

FILE_SCHEMA = "file://"
backend = "gloo"

def _dist_print(*msg, rank, delay=.5):
    time.sleep(delay * rank)
    print(f"rank {rank}:", *msg, flush=True)

def run(rank, world_size, init_method):
    torch.distributed.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

    dist_print = partial(_dist_print, rank=rank)
    dist_print(f"Rank {rank} of {world_size} initialized")

    torch.distributed.barrier()

    dist_print(f"Rank {rank} of {world_size} barrier passed")
    torch.distributed.destroy_process_group()

@contextmanager
def dist_context():
    # Get absolute path to avoid issues with relative paths
    rendezvous_file = os.path.abspath(filename)
    # Create the file *before* yielding the path
    open(rendezvous_file, "w").close()  # Or touch(filename)
    yield f"{FILE_SCHEMA}{rendezvous_file}"
    # No need to remove the file, torch.dist handles cleanup


if __name__ == "__main__":
    world_size = 2

    with dist_context() as rendezvous:
        mp.spawn(run, args=(world_size, rendezvous), nprocs=world_size, join=True)
