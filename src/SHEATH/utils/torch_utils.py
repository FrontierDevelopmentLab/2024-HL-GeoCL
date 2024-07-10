import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


def _float(tensor):
    return torch.Tensor(tensor.astype(np.float32)).float()
