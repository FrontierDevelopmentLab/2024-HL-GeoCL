import datetime as dt
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class GeoCLoakDataLoader(Dataset):
    def __init__(
        self, mapping_file, input_file, target_file_path, target_list, scaler_dir
    ):
        self.mapping_file = mapping_file
        self.input_file = input_file
        self.target_file_path = target_file_path
        self.target_list = target_list
        self.scaler_dir = scaler_dir
        self.mapping = pd.read_csv(self.mapping_file, parse_dates=[0, 1], header=0)
        self.input = pd.read_hdf(self.input_file, key="data")

        # Load existing scalers
        self.scaler_rtsw_pt, self.scaler_rtsw_std = self._load_rtsw_scaler(
            "scaler.json"
        )
        self.scaler_rtsw_pt._scaler = self.scaler_rtsw_std
        self.scaler_rtsw_pt.set_output(transform="pandas")
        self.scaler_db = {}
        self.target = {}
        for tar in target_list:
            self.target[tar + "_geo"] = pd.read_hdf(
                self.target_file_path + "/" + tar + "_all.h5", key="data"
            )
            self.scaler_db[tar + "_geo"] = self._load_db_scaler(
                "std_" + tar + "_scaler.json"
            )
            self.scaler_db[tar + "_geo"].set_output(transform="pandas")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        # read mapping
        input_ind = self.mapping.iloc[idx]["Time"]
        target_ind = self.mapping.iloc[idx]["Target_time"]
        input_df = self.input.loc[
            input_ind - dt.timedelta(minutes=90) : input_ind
        ].values
        target_dfs = []
        for tar in self.target_list:
            target_df = self.target[tar + "_geo"].loc[target_ind].values
            target_df = target_df.reshape(-1, 1)
            scaled_target_df = self.scaler_db[tar + "_geo"].transform(target_df.T)
            target_dfs.append(scaled_target_df)

        scaled_input_df = self.scaler_rtsw_pt.transform(input_df)
        concatenated_target_df = pd.concat(target_dfs, axis=1)
        mask_target = torch.tensor(
            ~np.isnan(concatenated_target_df).values.flatten(), dtype=torch.bool
        )
        input_data = torch.tensor(scaled_input_df.values, dtype=torch.float32)
        target_data = torch.tensor(
            concatenated_target_df.values.flatten(), dtype=torch.float32
        )
        # print(input_df.shape)
        kp_index = torch.tensor(input_df[-1, 9])

        return input_data, target_data, mask_target, kp_index

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


# # Example Use Case

# # Define paths
# mapping_file = ace_mapping_train_files[0]
# # input_path = ace_input_path
# # target_paths =  supermag_target_paths
# scaler_dir = scaler_dir

# # Initialize the GeoCLoakDataLoader with use_dbz_geo set to False
# geo_data_loader = GeoCLoakDataLoader(mapping_file, '/home/chetrajpandey/daggerdata/ace_all.h5',
#     '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)
# # start_time = time.time()
# # Create a PyTorch DataLoader
# batch_size = 2

# data_loader = DataLoader(geo_data_loader, batch_size=batch_size, num_workers=90, pin_memory=True, shuffle=True)
# # end_time = time.time()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Iterate through the data loader
# for epochs in range(1,2):
#     start_time = time.time()
#     for i, (inputs, targets, masks, kp) in enumerate(data_loader):
#         # start_time = time.time()
#         # worker_id = torch.utils.data.get_worker_info().id
#         inputs.to(device)
#         targets.to(device)
#         masks.to(device)
#         kp.to(device)
#         # inputs: Tensor of shape (batch_size, num_features)
#         # targets: Tensor of shape (batch_size, num_targets)
#         # masks: Tensor of shape (batch_size, num_targets) indicating valid target values
#         # print(worker_id)
#         print(f"Inputs: {inputs.shape}")
#         print(f"Targets: {targets.shape}")
#         print(f"Masks: {masks.shape}")
#         print(f"KP: {kp.shape}")
#         print("\n")


#         # print("time: ", end_time-start_time)

#         # del inputs, targets, masks
#         # torch.cuda.empty_cache()
#     end_time = time.time()
#     print(f'Epoch {epochs} Total Time', end_time-start_time)
