import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

def _float(tensor):
    return torch.Tensor(tensor.astype(np.float32)).float()