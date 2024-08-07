"""
This is the module for SHEATH which provides the dataloader,
MLP model, and other helper function for the same.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_scaler_from_json(filename):
    """
    Load a StandardScaler object from a JSON file.

    Parameters
    ----------
    filename: str
      Path to the JSON file containing the scaler.

    Returns
    -------
    scaler: StandardScaler
        The loaded scaler object.
    """
    with open(filename, "r") as f:
        scaler_dict = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_dict["mean"])
    scaler.scale_ = np.array(scaler_dict["scale"])
    scaler.var_ = np.array(scaler_dict["var"])
    return scaler


def calculate_metrics(predictions, targets, scaler_targets_dir):
    """
    Calculate RMSE, MAE, and R^2 metrics based on the inverse-transformed targets and predictions.

    Parameters
    ----------
    predictions: np.ndarray
      Model predictions (scaled).
    targets: np.ndarray
      True targets (scaled).
    scaler_targets_dir: str
      Directory path where scaler JSON is saved.

    Returns
    -------
    metrics: dict
      A dictionary containing RMSE, MAE, and R^2 scores.
    """
    # Load the target scaler
    scaler_targets = load_scaler_from_json(
        os.path.join(scaler_targets_dir, "scaler_targets.json")
    )

    # Inverse transform the targets and predictions
    targets_unscaled = scaler_targets.inverse_transform(targets)
    predictions_unscaled = scaler_targets.inverse_transform(predictions)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets_unscaled, predictions_unscaled))
    mae = mean_absolute_error(targets_unscaled, predictions_unscaled)
    r2 = r2_score(targets_unscaled, predictions_unscaled)

    return rmse, mae, r2


def calculate_individual_metrics(predictions, targets, scaler_targets_dir):
    """
    Calculate individual metrics for each target feature.

    Parameters
    ----------
    predictions: np.ndarray
      Model predictions (scaled).
    targets: np.ndarray
      True targets (scaled).
    scaler_targets_dir: str
      Directory path where scaler JSON is saved.

    Returns
    -------
    metrics: dict
      A dictionary containing individual RMSE, MAE, and R^2 scores for each feature.
    """
    # Load the target scaler
    scaler_targets = load_scaler_from_json(
        os.path.join(scaler_targets_dir, "scaler_targets.json")
    )

    # Inverse transform the targets and predictions
    targets_unscaled = scaler_targets.inverse_transform(targets)
    predictions_unscaled = scaler_targets.inverse_transform(predictions)

    # Calculate metrics for each feature
    individual_metrics = {}
    for i, feature_name in enumerate(
        ["Speed", "Density", "Temperature", "Bt", "Bx", "By", "Bz"]
    ):
        feature_rmse = np.sqrt(
            mean_squared_error(targets_unscaled[:, i], predictions_unscaled[:, i])
        )
        feature_mae = mean_absolute_error(
            targets_unscaled[:, i], predictions_unscaled[:, i]
        )
        feature_r2 = r2_score(targets_unscaled[:, i], predictions_unscaled[:, i])
        individual_metrics[f"{feature_name}_rmse"] = feature_rmse
        individual_metrics[f"{feature_name}_mae"] = feature_mae
        individual_metrics[f"{feature_name}_r2"] = feature_r2

    return individual_metrics


class DataProcessor:
    """
    A class to handle the processing of CSV data from a local directory.

    Attributes
    ----------
    directory: str
      The directory path where CSV files are located.
      column_names (list): List of column names from the merged DataFrame.

    Parameters
    ----------
    directory: str
      The directory path where CSV files are located.
    """

    def __init__(self, directory):
        """
        Initializes the DataProcessor with the specified directory.
        """
        self.directory = directory

    def read_and_merge_csvs(self):
        """
        Reads and merges CSV files from the specified directory.

        Returns:
            pd.DataFrame: A DataFrame containing the merged data from all CSV files.
        """
        csv_files = [f for f in os.listdir(self.directory) if f.endswith(".csv")]

        dfs = []
        for file in np.sort(csv_files):
            file_path = os.path.join(self.directory, file)
            df = pd.read_csv(file_path, low_memory=False)
            df = df.dropna()
            dfs.append(df)

        merged_df = pd.concat(dfs)
        self.column_names = merged_df.columns.tolist()
        return merged_df

    def create_splits(self, data, test_intervals, val_years_months):
        """
        Creates training, validation, and test sets from the given
        data based on specified intervals, ensuring that test and
        validation sets do not overlap.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data.
        test_interval: tuple
            A tuple of tuples, each containing the start and end dates for a test interval.
        val_years_months: dict
            A dictionary mapping years to months or tuples of start and end months for validation.

        Returns
        -------
        tuple: tuple
            A tuple containing the training, validation, and test sets.

        """
        initial_timestamp_col = self.column_names[1]
        data[initial_timestamp_col] = pd.to_datetime(data[initial_timestamp_col])

        # Initialize boolean Series for test and validation indices
        test_indices = pd.Series([False] * len(data), index=data.index)
        val_indices = pd.Series([False] * len(data), index=data.index)

        # Create test set mask
        for start, end in test_intervals:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            test_indices = test_indices | (
                (data[initial_timestamp_col] >= start_date)
                & (data[initial_timestamp_col] <= end_date)
            )

        test_set = data[test_indices]
        print("Test set length:", len(test_set))

        # Create validation set mask
        for key, value in val_years_months.items():
            if isinstance(value, tuple):
                start_month, end_month = value
                val_indices = val_indices | (
                    (data[initial_timestamp_col].dt.year == key)
                    & (data[initial_timestamp_col].dt.month >= start_month)
                    & (data[initial_timestamp_col].dt.month <= end_month)
                )
            else:
                val_indices = val_indices | (data[initial_timestamp_col].dt.year == key)

        # Ensure validation set does not overlap with test set
        val_indices = val_indices & ~test_indices
        val_set = data[val_indices]
        print("Validation set length:", len(val_set))

        # Create train set
        train_set = data[~test_indices & ~val_indices]
        print("Training set length:", len(train_set))

        return train_set, val_set, test_set

    def save_splits(self, train_set, val_set, test_set, output_directory):
        """
        Saves the training, validation, and test sets to CSV files in the specified directory.

        Parameters
        ----------
        train_set : pd.DataFrame
            The training data to be saved.
        val_set : pd.DataFrame
            The validation data to be saved.
        test_set : pd.DataFrame
            The directory where the CSV files will be saved.

        Returns
        -------
        None
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define the paths for the files in the directory
        train_path = os.path.join(output_directory, "sheath_train_set.csv")
        val_path = os.path.join(output_directory, "sheath_val_set.csv")
        test_path = os.path.join(output_directory, "sheath_test_set.csv")

        # Save DataFrames to CSV files
        train_set.to_csv(train_path, index=False)
        val_set.to_csv(val_path, index=False)
        test_set.to_csv(test_path, index=False)


class SHEATHDataLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing SHEATH data
    for training and testing, including applying scaling with StandardScaler.

    Attributes
    ----------
    directory: str
        Directory containing the data file.
    filename: str
      Name of the data file.
    scaler_dir: str
      Directory for saving or loading scalers.
    dataframe: pd.DataFrame
      Pandas DataFrame containing the loaded data.
    inputs: np.ndarray
      Scaled input features.
    targets: np.ndarray
      Scaled target variables.
    scaler_inputs : StandardScaler
      Standard Scaler for input features.
    scaler_targets: StandardScaler
      Scaler for target variables.


    Parameters
    ----------
    directory :str
      Directory containing the data file.
    filename: str
      Name of the data file.
    scaler_dir: str
      Directory for saving or loading scalers.
    is_train: bool
      Whether the dataset is used for training.
      If False, it is used for testing.
    """

    def __init__(self, directory, filename, scaler_dir, is_train=True):
        """
        Initializes the dataset by loading data from a CSV file, scaling it,
        and saving or loading scaler objects. Supports training and testing modes.
        """

        self.directory = directory
        self.filename = filename
        self.scaler_dir = scaler_dir
        self.is_train = is_train

        # Read the CSV data from the local directory
        file_path = os.path.join(directory, filename)
        self.dataframe = pd.read_csv(file_path)

        # Select input features and target variables
        self.inputs = self.dataframe.iloc[:, 2:-14]
        self.targets = self.dataframe.iloc[:, -14:]

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

    def _save_scaler(self, scaler, filename) -> None:
        """
        Saves a StandardScaler object to a JSON file.

        Parameters
        ----------
        scaler: StandardScaler
          The scaler object to save.
        filename: str
          Name of the JSON file to save the scaler.

        Returns
        -------
        None
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

    def _load_scaler(self, filename):
        """
        Loads a StandardScaler object from a JSON file.

        Parameters
        ----------
        filename :str
          Name of the JSON file containing the scaler.

        Returns
        -------
        scalar: StandardScaler
           The loaded scaler object.
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

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Parameters
        ----------
        idx :int
            Index of the sample to retrieve.

        Returns
        -------
        tuple: A tuple containing:
            - input_data (torch.Tensor): Scaled input features.
            - target_data (torch.Tensor): Scaled target data.
        """
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_data, target_data


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  MACHINE LEARNING MODEL


class SHEATH_MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout_rate=0.3, init_type="kaiming"
    ):
        super(SHEATH_MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Apply the chosen initialization method
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        """
        Intialize the weights of the MLP model.

        Parameters
        ----------
        init_type : str
            The intialization model.
        """
        if init_type == "kaiming":
            init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
            init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
            init.kaiming_normal_(self.fc3.weight, mode="fan_in", nonlinearity="relu")
        elif init_type == "xavier":
            init.xavier_normal_(self.fc1.weight)
            init.xavier_normal_(self.fc2.weight)
            init.xavier_normal_(self.fc3.weight)
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

    def forward(self, x):
        """
        The core ML model.
        """
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
