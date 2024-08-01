# imports
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import PowerTransformer, StandardScaler
import json
import os

class GeoCLoakDataLoader(Dataset):
    def __init__(self, mapping_file, input_path, target_paths,scaler_dir):
        self.mapping_file = mapping_file
        self.input_path = input_path
        self.target_paths = target_paths  # List of target paths
        self.scaler_dir = scaler_dir
        self.mapping = pd.read_csv(self.mapping_file, parse_dates=[0, 1], header=0)
        
        # Load existing scalers
        self.scaler_rtsw_pt, self.scaler_rtsw_std = self._load_rtsw_scaler("scaler.json")
        self.scaler_rtsw_pt._scaler = self.scaler_rtsw_std
        self.scaler_rtsw_pt.set_output(transform='pandas')
        self.scaler_dbe = self._load_db_scaler("dbe_scaler.json")
        self.scaler_dbe.set_output(transform='pandas')
        self.scaler_dbn = self._load_db_scaler("dbn_scaler.json")
        self.scaler_dbn.set_output(transform='pandas')
        self.scaler_dbz = self._load_db_scaler("dbz_scaler.json")
        self.scaler_dbz.set_output(transform='pandas')
           
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
        scaled_input_df = self.scaler_rtsw_pt.transform(input_df)
          
        target_dfs = []
        for target_path in self.target_paths:
            full_target_path = path.join(target_path, target_filename)
            if not path.exists(full_target_path):
                raise FileNotFoundError(f"Target file {target_filename} not found in {target_path}.")
            target_df = pd.read_csv(full_target_path, header=None)
            if target_path.split('/')[-1] == 'dbe_geo':
                scaled_target_df = self.scaler_dbe.transform(target_df)
            elif target_path.split('/')[-1] == 'dbn_geo':
                scaled_target_df = self.scaler_dbn.transform(target_df)
            elif target_path.split('/')[-1] == 'dbz_geo':
                scaled_target_df = self.scaler_dbz.transform(target_df)
            else:
                raise ValueError(f"Unknown target path {target_path}.")
            target_dfs.append(scaled_target_df)
        
        concatenated_target_df = pd.concat(target_dfs, axis=1)
        
        mask_target = torch.tensor(~np.isnan(concatenated_target_df).values.flatten(), dtype=torch.bool)
        
        input_data = torch.tensor(scaled_input_df.values, dtype=torch.float32)
        target_data = torch.tensor(concatenated_target_df.values.flatten(), dtype=torch.float32)
        
        return input_data, target_data, mask_target

    def _save_rtsw_scaler(self, scaler, filename):
        """
        Saves a PowerTransfomer(method='yeo-johnson', standardize=True) object to a JSON file.

        Args:
            scaler (PowerTransfomer): The scaler object to save.
            filename (str): Name of the JSON file to save the scaler.
        """
        
        pt_scaler_dict = {
            "params": scaler.get_params(),
            "lambdas": scaler.lambdas_.tolist(),
            "n_features": scaler.n_features_in_,
            "feature_names": scaler.feature_names_in_.tolist(),
        }
        std_scaler_dict = {
            "mean": scaler._scaler.mean_.tolist(),
            "scale": scaler._scaler.scale_.tolist(),
            "var": scaler._scaler.var_.tolist(),
        }
        with open(os.path.join(self.scaler_dir, 'pt_'+filename), "w") as f:
            json.dump(pt_scaler_dict, f)
            
        with open(os.path.join(self.scaler_dir, 'std_'+filename), "w") as f:
            json.dump(std_scaler_dict, f)

    def _save_db_scaler(self, scaler, filename):
        """
        Saves a StandardScaler object to a JSON file.

        Args:
            scaler (StandardScaler): The scaler object to save.
            filename (str): Name of the JSON file to save the scaler.
        """
        scaler_dict = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "n_features": scaler.n_features_in_,
            "feature_names": scaler.feature_names_in_.tolist(),
        }
        with open(os.path.join(self.scaler_dir, filename), "w") as f:
            json.dump(scaler_dict, f)

    def _load_rtsw_scaler(self, filename):
        """
        Loads a PowerTransformer and associated StandardScaler object from JSON files.

        Args:
            filename (str): Name of the JSON file containing the scaler.

        Returns:
            PowerTransformer: The loaded scaler object.
        """
        
        with open(os.path.join(self.scaler_dir, 'pt_'+filename), "r") as f:
            pt_scaler_dict = json.load(f)
            
        pt_scaler = PowerTransformer(method='yeo-johnson', standardize=False)
        
        for key, value in pt_scaler_dict["params"].items():
            pt_scaler.set_params(**{key: value})
            
        pt_scaler.lambdas_ = np.array(pt_scaler_dict["lambdas"])
        pt_scaler.n_features_in_ = pt_scaler_dict["n_features"]
        pt_scaler.feature_names_in_ = np.array(pt_scaler_dict["feature_names"])
        
        with open(os.path.join(self.scaler_dir, 'std_'+filename), "r") as f:
            std_scaler_dict = json.load(f)
            
        std_scaler = StandardScaler()
        
        std_scaler.mean_ = np.array(std_scaler_dict["mean"])
        std_scaler.scale_ = np.array(std_scaler_dict["scale"])
        std_scaler.var_ = np.array(std_scaler_dict["var"])
                
        return pt_scaler, std_scaler
    
    def _load_db_scaler(self, filename):
        """
        Loads a StandardScaler object from a JSON file.

        Args:
            filename (str): Name of the JSON file containing the scaler.

        Returns:
            StandardScaler: The loaded scaler object.
        """
        with open(os.path.join(self.scaler_dir, filename), "r") as f:
            scaler_dict = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_dict["mean"])
        scaler.scale_ = np.array(scaler_dict["scale"])
        scaler.var_ = np.array(scaler_dict["var"])
        scaler.n_features_in_ = scaler_dict["n_features"]
        scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])
        return scaler
