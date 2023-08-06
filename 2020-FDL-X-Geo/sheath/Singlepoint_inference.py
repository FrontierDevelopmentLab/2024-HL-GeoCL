import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import pickle

from SHEATH_atomic_predict import CloudFetcher, AtomicSamurai
from SHEATH_module import SHEATH, HSENN
from utils.preprocessing import Preprocessor_CH
from utils.scalers import AIA_StandardScaler


# -----
model_hyperparams = pickle.load(open('logs/model_hyperparams.scaler', 'rb'))
n_passbands = model_hyperparams['n_passbands'] 
height = model_hyperparams['height'] 
width = model_hyperparams['width'] 
n_out = model_hyperparams['n_out'] 
# -------
# Define the cloud fetcher object to get the data
cloudfetcher_object = CloudFetcher()

# Define the scaler
scales = pickle.load(open("logs/scaler_aia.scaler",'rb'))
scaler_aia = AIA_StandardScaler(mean = scales['mean'], stddev = scales['stddev'])

# Define the preprocessing module
preprocessor = Preprocessor_CH(npix = 17, n_size = 512, 
                               ch_mask = True, scaler_aia = scaler_aia)


#Define the NN model
HSE_model = HSENN(n_passbands = n_passbands,height = height,
                      width = width, n_out = n_out)
HSE_model.load_state_dict(torch.load("logs/SHEATH.ckpt"))

# Load the solar wind variable names and scalers
sw_vars_list = pickle.load(open("logs/sw_variables.pickle",'rb'))
scaler_y = pickle.load(open("logs/scaler_y.scaler",'rb'))

# Instantiate the SHEATH module
sheathobj = SHEATH(preprocessor, HSE_model, sw_vars_list, scaler_y)

# Instantiate the atomic inference module
atom = AtomicSamurai(cloudfetcher_object, sheathobj)
sw_values = atom.atomic_inference(aia_timestamp = "2010-08-20 00:20:00", aia_idx = 500, 
                                  hmi_timestamp = "2010-08-20 00:20:00", hmi_idx = 500)

print(sw_values)