# imports
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import pandas as pd
import numpy as np
from io import StringIO
from os import path

class GeoCLoakDataLoader(Dataset):
    def __init__(self, mapping_file, bucket_name, input_path, target_path):
        
        self.mapping_file = mapping_file
        self.bucket = bucket_name
        self.input_path = input_path
        self.target_path = target_path
        
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket)
        mapping_subfolder,mapping_fnm = path.split(self.mapping_file)
        mappings_blob = self._get_file_from_subfolder(mapping_fnm,mapping_subfolder)
        self.mapping = pd.read_csv(StringIO(mappings_blob), parse_dates=[0,1], header=0)
        
    def __len__(self):
        return 10 #len(self.mapping)
    
    def __getitem__(self, idx):
        input_timestamp = self.mapping.iloc[idx, 0]
        target_timestamp = self.mapping.iloc[idx, 1]
        
        input_filename = input_timestamp.strftime("%Y%m%d%H%M") + ".csv"
        target_filename = target_timestamp.strftime("%Y%m%d%H%M") + ".csv"
        
        # need to merge dbe and den into targets
        
        input_blob = self._get_file_from_subfolder(input_filename,self.input_path)
        target_blob = self._get_file_from_subfolder(target_filename,self.target_path)
        
        input_df = pd.read_csv(StringIO(input_blob), header=None)
        target_df = pd.read_csv(StringIO(target_blob), header=None)
        
        mask_target = torch.tensor(~np.isnan(target_df).values.flatten(), dtype=torch.bool)
        
        input_data = torch.tensor(input_df.values, dtype=torch.float32)
        target_data = torch.tensor(target_df.values.flatten(), dtype=torch.float32)
        
        # add scalar file stuff
        
        return input_data, target_data, mask_target
 
    def _get_file_from_subfolder(self, filename, subfolder):
        """
        Retrieve a file from a GCS bucket subfolder.
        
        Args:
        - filename (str): Filename to search for.
        - subfolder (str): Subfolder path in the bucket.
        
        Returns:
        - file_content (str): File content as a string.
        """
        blob_name = f"{subfolder}/{filename}"
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            return blob.download_as_text()
        else:
            raise FileNotFoundError(f"File {filename} not found in subfolder {subfolder}.")
    
   
ds = GeoCLoakDataLoader('formatted_data/ACE/ace_processed/ace_mapping.csv','geocloak2024','formatted_data/ACE/ace_processed/ace_processed','formatted_data/SuperMAG/supermag_processed/dbn_geo')
dl = DataLoader(ds, batch_size=3, shuffle=True)

# display dataloader output

for i, (input_data, target_data, mask_target) in enumerate(dl):
    print(f"Batch {i+1}")
    print(f"Input data shape: {input_data.shape}")
    print(f"Target data shape: {target_data.shape}")
    print(f"Mask target shape: {mask_target.shape}")
    print("")