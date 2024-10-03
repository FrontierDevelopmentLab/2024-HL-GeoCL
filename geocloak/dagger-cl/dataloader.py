# imports

import json
import os
import warnings
from os import path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
import time

from paths import (
    ace_input_path,
    mapping_test_file,
    mapping_train_files,
    mapping_val_file,
    scaler_dir,
    supermag_target_paths,
)


class GeoCLoakDataLoader(Dataset):
    def __init__(self, mapping_file, input_path, target_paths, scaler_dir):
        self.mapping_file = mapping_file
        self.input_path = input_path
        self.target_paths = target_paths  # List of target paths
        self.scaler_dir = scaler_dir
        self.mapping = pd.read_csv(self.mapping_file, parse_dates=[0, 1], header=0)
        self.input_fnms = self.mapping.iloc[:, 0].apply(
            lambda x: x.strftime("%Y%m%d%H%M") + ".csv"
        )
        self.target_fnms = self.mapping.iloc[:, 1].apply(
            lambda x: x.strftime("%Y%m%d%H%M") + ".csv"
        )

        # Load existing scalers
        self.scaler_rtsw_pt, self.scaler_rtsw_std = self._load_rtsw_scaler(
            "scaler.json"
        )
        self.scaler_rtsw_pt._scaler = self.scaler_rtsw_std
        self.scaler_rtsw_pt.set_output(transform="pandas")
        self.scaler_db = {}
        self.scaler_db["dbe_geo"] = self._load_db_scaler("std_dbe_scaler.json")
        self.scaler_db["dbe_geo"].set_output(transform="pandas")
        self.scaler_db["dbn_geo"] = self._load_db_scaler("std_dbn_scaler.json")
        self.scaler_db["dbn_geo"].set_output(transform="pandas")
        self.scaler_db["dbz_geo"] = self._load_db_scaler("std_dbz_scaler.json")
        self.scaler_db["dbz_geo"].set_output(transform="pandas")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        input_filename = self.input_fnms[idx]
        target_filename = self.target_fnms[idx]
        input_path = path.join(self.input_path, input_filename)
        if not path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_filename} not found.")
        input_df = pd.read_csv(
            input_path, header=None, engine="c", memory_map=True, dtype=np.float32
        )
        scaled_input_df = self.scaler_rtsw_pt.transform(input_df)
        target_dfs = []
        for target_path in self.target_paths:
            comp = target_path.split("/")[-1]
            full_target_path = path.join(target_path, target_filename)
            if not path.exists(full_target_path):
                raise FileNotFoundError(
                    f"Target file {target_filename} not found in {target_path}."
                )
            target_df = pd.read_csv(
                full_target_path,
                header=None,
                engine="c",
                memory_map=True,
                dtype=np.float32,
            )
            print(target_df.shape)
            scaled_target_df = self.scaler_db[comp].transform(target_df.T)
            target_dfs.append(scaled_target_df)
        concatenated_target_df = pd.concat(target_dfs, axis=1)
        mask_target = torch.tensor(
            ~np.isnan(concatenated_target_df).values.flatten(), dtype=torch.bool
        )
        input_data = torch.tensor(scaled_input_df.values, dtype=torch.float32)
        target_data = torch.tensor(
            concatenated_target_df.values.flatten(), dtype=torch.float32
        )

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
        with open(os.path.join(self.scaler_dir, "pt_" + filename), "w") as f:
            json.dump(pt_scaler_dict, f)

        with open(os.path.join(self.scaler_dir, "std_" + filename), "w") as f:
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

        with open(os.path.join(self.scaler_dir, "pt_" + filename), "r") as f:
            pt_scaler_dict = json.load(f)

        pt_scaler = PowerTransformer(method="yeo-johnson", standardize=False)

        for key, value in pt_scaler_dict["params"].items():
            pt_scaler.set_params(**{key: value})

        pt_scaler.lambdas_ = np.array(pt_scaler_dict["lambdas"])
        pt_scaler.n_features_in_ = pt_scaler_dict["n_features"]
        pt_scaler.feature_names_in_ = np.array(pt_scaler_dict["feature_names"])

        with open(os.path.join(self.scaler_dir, "std_" + filename), "r") as f:
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


# Example Use Case

# Define paths
mapping_file = mapping_train_files[0]
input_path = ace_input_path
target_paths = supermag_target_paths
scaler_dir = scaler_dir

# Initialize the GeoCLoakDataLoader with use_dbz_geo set to False
geo_data_loader = GeoCLoakDataLoader(
    mapping_file=mapping_file,
    input_path=input_path,
    target_paths=target_paths,
    scaler_dir=scaler_dir,
)
# start_time = time.time()
# Create a PyTorch DataLoader
batch_size = 2048

data_loader = DataLoader(
    geo_data_loader,
    batch_size=batch_size,
    num_workers=80,
    pin_memory=True,
    shuffle=True,
)
# end_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Iterate through the data loader
for epochs in range(1, 2):
    start_time = time.time()
    for i, (inputs, targets, masks) in enumerate(data_loader):
        # start_time = time.time()
        # worker_id = torch.utils.data.get_worker_info().id
        inputs.to(device)
        targets.to(device)
        masks.to(device)
        # inputs: Tensor of shape (batch_size, num_features)
        # targets: Tensor of shape (batch_size, num_targets)
        # masks: Tensor of shape (batch_size, num_targets) indicating valid target values
        # print(worker_id)
        print(f"Inputs: {inputs.shape}")
        print(f"Targets: {targets.shape}")
        print(f"Masks: {masks.shape}")
        print("\n")

        # print("time: ", end_time-start_time)

        # del inputs, targets, masks
        # torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Epoch {epochs} Total Time", end_time - start_time)
