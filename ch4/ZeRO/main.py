import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def print_peak_mem(prefix, device):
    if device == 0: # Only print from rank 0
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6} MB")


def example(rank, world_size, use_zero):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
   
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(10)])
    print_peak_mem("Max memory after model creation", rank)

    ddp_model = DDP(model, device_ids=[rank]) # Wrap model in DDP
    print_peak_mem("Max memory after DDP", rank)

    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=optim.Adam,
            lr = 0.01
        )
    else:
        optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)

    outputs = ddp_model(torch.randn(20, 2000).to(rank))
    labels = torch.randn(20, 2000).to(rank)

    loss_fn(outputs, labels).backward()

    print_peak_mem("Max memory after backward", rank)
    optimizer.step()
    print_peak_mem("Max memory after optimizer step", rank)

    print(f"params sum: {sum(model.parameters()).sum()}")


def main():
    world_size = 2
    print("=== Using ZeRO ===")
    mp.spawn( # spawn creates multiple processes
        example, # example function will be called
        args=(world_size, True), # args for example function
        nprocs=world_size, # number of processes to spawn
        join=True # wait for all processes to finish
    )

    print("=== Without ZeRO ===")
    mp.spawn(
        example,
        args=(world_size, False),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()