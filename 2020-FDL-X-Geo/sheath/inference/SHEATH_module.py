"""
 In this script we define functions and modules of sheath which will perform the preprocessing for SHEATH.
"""
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import numpy as np
from utils.torch_utils import _float
import pandas as pd

class HSENN(nn.Module):
    """
        This is the Neural network-based pytorch model which will perform inference given a 
        processed dataset. 
    """
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
    
    def predict(self,aiadata,**kwargs):
        with torch.no_grad():
            logits = self.linear(self.model(aiadata.permute(0,3,1,2)))
        return logits.detach().cpu().numpy()
    

class SHEATH:
    """
        This is the main SHEATH module. This will take in the model, the data, and the preprocesser 
        to perform inferences. 
    """
    def __init__(self, preprocessor, model, solarwind_param_list, scaler_y = None):
        """
            Preprocess is a function which will be called to preprocess aia and hmi data.
            Model will perform inferences.
        """
        self.preprocessor = preprocessor
        self.model = model
        self.solarwind_param_list = solarwind_param_list #List of solar wind parameters
        self.scaler_y = scaler_y
    def predict(self,inputdatacube):
        # Expect a numpy array with various solar wind parameters here.
        model_output = self.model.predict(inputdatacube)
        if self.scaler_y is not None:
            model_output = self.scaler_y.inverse_transform(model_output)
        solar_wind = pd.DataFrame.from_dict({param: model_output[:,i] for i, param in enumerate(self.solarwind_param_list)})
        return solar_wind
        
    
    
        
    