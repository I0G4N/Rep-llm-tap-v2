import torch
from torch.utils.data import Sampler
import torch.utils.data.distributed as dist
import math

class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() # set number of replicas(processes) based on world size(total processes)
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() # set rank(process id)
        self.dataset = dataset # dataset to be sampled
        self.num_replicas = num_replicas
        self.rank = rank # rank of the current process(GPU)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) # number of samples per process
        self.total_size = self.num_samples * self.num_replicas # total size of the dataset across all processes
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator() # create a random number generator
            g.manual_seed(self.seed + self.epoch) # every epoch should have a different shuffle, let different processes have different seeds
            indices = torch.randperm(len(self.dataset), generator=g).tolist() # random permutation of indices
        else:
            indices = list(range(len(self.dataset))) # sequential indices

        # when the dataset size is not divisible by the number of processes, we need to pad the indices
        # so that each process has the same number of samples
        # it is same as padding the dataset with the first few elements
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, "Length of indices should be equal to total size"

        indices = indices[self.rank:self.total_size:self.num_replicas] # select indices for the current process
        assert len(indices) == self.num_samples, "Length of indices should be equal to number of samples"

        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        # when the epoch changes, we need to change the seed for shuffling
        self.epoch = epoch

        