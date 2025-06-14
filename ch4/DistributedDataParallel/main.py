import argparse
import os
import shutil
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.distributed import DistributedSampler
from torchvision import transforms

from model.models import DeepLab
from utils.dataset import Cityscaples

parser = argparse.ArgumentParser(description='DeepLab')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)') # metavar means the name of the argument in usage messages
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=3, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR',)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local rank for DistributedDataParallel')

args = parser.parse_args()

torch.distributed.init_process_group(backend='nccl') # init, nccl is the recommended backend for distributed training on NVIDIA GPUs
args.local_rank = int(os.environ['LOCAL_RANK']) # get local rank from environment variable
print("Use GPU: {} for training".format(args.local_rank))

model = DeepLab()

torch.cuda.set_device(args.local_rank) # set the device for the current process
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                  output_device=args.local_rank, find_unused_parameters=True)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

image_transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

label_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
])

train_dataset = Cityscaples(root='data/cityscapes', split='train', transform=image_transform, target_transform=label_transform)
train_sampler = DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,  # Shuffle is handled by DistributedSampler
    num_workers=args.workers,
    pin_memory=True, # Pin memory for faster data transfer to GPU
    sampler=train_sampler
)

for epoch in range(args.start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

if args.local_rank == 0:  # Only save the model from the main process
    torch.save(model.state_dict(), 'model.pth')
