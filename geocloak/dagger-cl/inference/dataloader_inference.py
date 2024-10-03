import json
import os
import time
import warnings
from os import path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class GeoCLoakDataLoader(Dataset):
    def __init__(self, mapping_file, input_path, scaler_dir):
        self.mapping_file = mapping_file
        self.input_path = input_path
        self.scaler_dir = scaler_dir
        self.mapping = pd.read_csv(self.mapping_file, parse_dates=[0], header=0)
        self.input_fnms = self.mapping.iloc[:, 0].apply(
            lambda x: x.strftime("%Y%m%d%H%M") + ".csv"
        )

        # Load scalers
        self.scaler_rtsw_pt, self.scaler_rtsw_std = self._load_rtsw_scaler(
            "scaler.json"
        )
        self.scaler_rtsw_pt._scaler = self.scaler_rtsw_std

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        input_filename = self.input_fnms[idx]
        input_path = path.join(self.input_path, input_filename)
        if not path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_filename} not found.")

        input_df = pd.read_csv(
            input_path, header=None, engine="c", memory_map=True, dtype=np.float32
        )
        scaled_input_df = self.scaler_rtsw_pt.transform(input_df)

        input_data = torch.tensor(scaled_input_df, dtype=torch.float32)
        return input_data, str(input_filename)

    def _load_rtsw_scaler(self, filename):
        with open(os.path.join(self.scaler_dir, "pt_" + filename), "r") as f:
            pt_scaler_dict = json.load(f)

        pt_scaler = PowerTransformer(method="yeo-johnson", standardize=False)
        pt_scaler.set_params(**pt_scaler_dict["params"])
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


# Initialize the GeoCLoakDataLoader with use_dbz_geo set to False
geo_data_loader = GeoCLoakDataLoader(
    mapping_file="/home/chetrajpandey/data/test_samples/ace_test_sample.csv",
    input_path="/home/jupyter/ace_processed/ace_processed",
    scaler_dir="/home/jupyter/models/DAGGER_CL",
)
# start_time = time.time()
# Create a PyTorch DataLoader
batch_size = 1

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
    for i, (inputs, filename) in enumerate(data_loader):
        # start_time = time.time()
        # worker_id = torch.utils.data.get_worker_info().id
        inputs.to(device)

        # inputs: Tensor of shape (batch_size, num_features)
        # targets: Tensor of shape (batch_size, num_targets)
        # masks: Tensor of shape (batch_size, num_targets) indicating valid target values
        # print(worker_id)
        print(f"Inputs: {inputs.shape}")
        print(f"Filename: {filename}")
        print("\n")

        # print("time: ", end_time-start_time)

        # del inputs, targets, masks
        # torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Epoch {epochs} Total Time", end_time - start_time)
