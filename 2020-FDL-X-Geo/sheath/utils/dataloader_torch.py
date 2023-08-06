import torch
import torch.nn as nn
from torch.utils import data
from utils.torch_utils import _float

class Data(data.Dataset):
    def __init__(self,features,target):
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = _float(features).to(device)
        self.target = _float(target).to(device)
        self.device = device
        self.nout = target.shape[-1]
        self.height = features.shape[1]
        self.width = features.shape[2]
        self.n_passbands= features.shape[3]
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self,index):
        return self.features[index],self.target[index]