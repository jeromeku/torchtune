import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size, init_method):
    # Initialize the process group using the file:// init_method.
    dist.init_process_group(
        backend="gloo",  # or "nccl" if GPUs are available
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    print(f"Process with rank {rank} has joined the process group.")
    
    # Perform a simple operation (e.g., a barrier) to synchronize.
    dist.barrier()
    print(f"Process with rank {rank} passed the barrier.")
    
    # Cleanup the process group.
    dist.destroy_process_group()

def main():
    world_size = 2
    
    # Create a temporary file that will be used as the rendezvous point.
    with tempfile.NamedTemporaryFile() as tmp:
        init_method = f"file://{tmp.name}"
        print(f"Using init_method: {init_method}")
        
        # Manually spawn processes; each process will be assigned a rank.
        mp.spawn(run, args=(world_size, init_method), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
