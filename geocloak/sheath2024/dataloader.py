import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class SHEATHDataLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing SHEATH data
    for training and testing, including applying scaling with StandardScaler.

    Attributes:
        directory (str): Directory containing the data file.
        filename (str): Name of the data file.
        scaler_dir (str): Directory for saving or loading scalers.
        dataframe (pd.DataFrame): DataFrame containing the loaded data.
        inputs (np.ndarray): Scaled input features.
        targets (np.ndarray): Scaled target variables.
        scaler_inputs (StandardScaler): Scaler for input features.
        scaler_targets (StandardScaler): Scaler for target variables.
    """

    def __init__(self, directory, filename, scaler_dir, is_train=True):
        """
        Initializes the dataset by loading data from a CSV file, scaling it,
        and saving or loading scaler objects. Supports training and testing modes.

        Args:
            directory (str): Directory containing the data file.
            filename (str): Name of the data file.
            scaler_dir (str): Directory for saving or loading scalers.
            is_train (bool): Whether the dataset is used for training.
                             If False, it is used for testing.
        """
        self.directory = directory
        self.filename = filename
        self.scaler_dir = scaler_dir
        self.is_train = is_train

        # Read the CSV data from the local directory
        file_path = os.path.join(directory, filename)
        self.dataframe = pd.read_csv(file_path)

        # Select input features and target variables
        self.inputs = self.dataframe.iloc[:, 2:-7].values
        self.targets = self.dataframe.iloc[:, -7:].values

        # Initialize scalers and apply them
        if self.is_train:
            # For training data: fit and save scalers
            self.scaler_inputs = StandardScaler()
            self.scaler_targets = StandardScaler()
            self.inputs = self.scaler_inputs.fit_transform(self.inputs)
            self.targets = self.scaler_targets.fit_transform(self.targets)

            # Save scalers as JSON
            os.makedirs(self.scaler_dir, exist_ok=True)
            self._save_scaler(self.scaler_inputs, "scaler_inputs.json")
            self._save_scaler(self.scaler_targets, "scaler_targets.json")
        else:
            # For testing data: load and apply existing scalers
            self.scaler_inputs = self._load_scaler("scaler_inputs.json")
            self.scaler_targets = self._load_scaler("scaler_targets.json")
            self.inputs = self.scaler_inputs.transform(self.inputs)
            self.targets = self.scaler_targets.transform(self.targets)

    def _save_scaler(self, scaler, filename):
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
        }
        with open(os.path.join(self.scaler_dir, filename), "w") as f:
            json.dump(scaler_dict, f)

    def _load_scaler(self, filename):
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
        return scaler

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - input_data (torch.Tensor): Scaled input features.
                - target_data (torch.Tensor): Scaled target data.
        """
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_data, target_data


# """ USE CASE"""
# def main():
#     # Specifying the local directory
#     directory = "/home/chetrajpandey/data/formatted_data/sheath_splits"
#     scaler_dir = "/home/chetrajpandey/data/formatted_data/sheath_splits"

#     # Create datasets
#     train_dataset = SHEATHDataLoader(directory, 'train_set.csv', scaler_dir, is_train=True)
#     val_dataset = SHEATHDataLoader(directory, 'val_set.csv', scaler_dir, is_train=False)
#     test_dataset = SHEATHDataLoader(directory, 'test_set.csv', scaler_dir, is_train=False)

#     # Print the number of samples in each dataset
#     print(f"Training set length: {len(train_dataset)}")
#     print(f"Validation set length: {len(val_dataset)}")
#     print(f"Test set length: {len(test_dataset)}")

#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = main()
#     for i, (inp, tar) in enumerate(test_loader):
#         print(inp.shape, tar.shape, inp)
#         break
