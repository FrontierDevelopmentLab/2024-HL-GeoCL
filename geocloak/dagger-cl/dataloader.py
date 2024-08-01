# imports
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from os import path

class GeoCLoakDataLoader(Dataset):
    def __init__(self, mapping_file, input_path, target_paths):
        self.mapping_file = mapping_file
        self.input_path = input_path
        self.target_paths = target_paths  # List of target paths
        
        self.mapping = pd.read_csv(self.mapping_file, parse_dates=[0, 1], header=0)
        
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        input_timestamp = self.mapping.iloc[idx, 0]
        target_timestamp = self.mapping.iloc[idx, 1]
        
        input_filename = input_timestamp.strftime("%Y%m%d%H%M") + ".csv"
        target_filename = target_timestamp.strftime("%Y%m%d%H%M") + ".csv"
        
        input_path = path.join(self.input_path, input_filename)
        
        if not path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_filename} not found.")
        
        input_df = pd.read_csv(input_path, header=None)
        
        target_dfs = []
        for target_path in self.target_paths:
            full_target_path = path.join(target_path, target_filename)
            if not path.exists(full_target_path):
                raise FileNotFoundError(f"Target file {target_filename} not found in {target_path}.")
            target_df = pd.read_csv(full_target_path, header=None)
            target_dfs.append(target_df)
        
        concatenated_target_df = pd.concat(target_dfs, axis=1)
        
        mask_target = torch.tensor(~np.isnan(concatenated_target_df).values.flatten(), dtype=torch.bool)
        
        input_data = torch.tensor(input_df.values, dtype=torch.float32)
        target_data = torch.tensor(concatenated_target_df.values.flatten(), dtype=torch.float32)
        
        return input_data, target_data, mask_target

