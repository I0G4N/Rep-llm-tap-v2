import os
from typing import Callable

import torch
import torch.distributed as dist

def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"): # fn is a callable that takes rank and size
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


import torch.multiprocessing as mp

def do_all_reduce(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group) # reduce operation dist.ReduceOp.[SUM, PRODUCT, MIN, MAX]
    # all_gather, gather, scatter, broadcast, reduce_scatter
    print(f"[{rank}] data = {tensor[0]}")


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")  # Use spawn to create processes
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_all_reduce))
        p.start() # Start the process
        processes.append(p)

    for p in processes:
        p.join() # Wait for all processes to finish
    print("All processes have completed.")