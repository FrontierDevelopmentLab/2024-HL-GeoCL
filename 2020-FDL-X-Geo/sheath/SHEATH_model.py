import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import numpy as np



def _float(tensor):
    return torch.Tensor(tensor.astype(np.float32)).float()


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


class HSENN(nn.Module):
    def __init__(self,**kwargs):
        super(HSENN,self).__init__()
        n_passbands = kwargs.pop('n_passbands',None)
        height = kwargs.pop('height',None)
        width = kwargs.pop('width',None)
        n_out = kwargs.pop('n_out',None)
        dropout_prob = kwargs.pop('dropout',0.3)
        if n_passbands is None or height is None or n_out is None or width is None:
            raise ValueError("Number of input passbands (filters) must be given!")
            
        # Define some model here
        self.model = nn.Sequential(
                    nn.Conv2d(n_passbands,13,kernel_size=(13,3),stride=1,dilation=3),
                    nn.MaxPool2d(kernel_size=(3,1)),
                    nn.ELU(),
                    nn.Conv2d(13,15,kernel_size=(9,3),stride=1,dilation=1),
                    nn.MaxPool2d(kernel_size=(3,1)),
                    nn.ELU(),
                    nn.Conv2d(15,17,kernel_size=(7,1),stride=1,dilation=1),
                    nn.MaxPool2d(kernel_size=(3,1)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
        dummy = torch.rand(1,n_passbands,height,width)
        dummy_forward = self.model(dummy)
        linear_shape = dummy_forward.shape[-1]
        self.linear = nn.Sequential(
                    nn.Linear(linear_shape,9),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_prob),
                    nn.Linear(9,n_out),
                    nn.ReLU()
                )

    def forward(self,aiadata,**kwargs):
        # Call the model and do forward pass.
        logits = self.linear(self.model(aiadata.permute(0,3,1,2)))
        return logits
