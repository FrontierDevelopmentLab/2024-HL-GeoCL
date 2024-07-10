import numpy as np
import torch


def _float(tensor):
    return torch.Tensor(tensor.astype(np.float32)).float()
