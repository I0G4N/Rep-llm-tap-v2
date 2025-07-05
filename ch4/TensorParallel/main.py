import torch
from torch.distributed._tensor import DeviceMesh, Replicate, Shard, distributed_tensor, distributed_module
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x) + self.fc2(x))
    

mesh = DeviceMesh("cuda", [[0, 1], [2, 3]]) # 2x2 device mesh, distributed by rows

def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0), Replicate()] # shard by rows, replicate across columns
    if mod_name == "fc1":
        mod.weight = torch.nn.Parameter(
            distributed_tensor(mod.weight, mesh, rowwise_placement)
        )
    
sharded_module = distributed_module(MyModel(), mesh, shard_fc)
