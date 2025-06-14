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

from model.models import DeepLab
from utils.dataset import Cityscaples

parser = argparse.ArgumentParser(description='DeepLab')


