import json
import os
import sys
import time

import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import tqdm as tq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("../"))
# from magalert.models.sheath import calculate_individual_metrics, calculate_metrics
# from magalert.preprocess.backtrack import ballistic


def load_scaler_from_json(filename, scaler_type="standard"):
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
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_dict["mean"])
        scaler.scale_ = np.array(scaler_dict["scale"])
        scaler.var_ = np.array(scaler_dict["var"])
        scaler.n_features_in_ = scaler_dict["n_features"]
        scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=tuple(scaler_dict["feature_range"]))
        scaler.min_ = np.array(scaler_dict["min"])
        scaler.scale_ = np.array(scaler_dict["scale"])
        scaler.data_min_ = np.array(scaler_dict["data_min"])
        scaler.data_max_ = np.array(scaler_dict["data_max"])
        scaler.n_features_in_ = scaler_dict["n_features"]
        scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])

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
        os.path.join(scaler_targets_dir, "scaler_targets.json"),
        scaler_type="minmax",
    )

    # Inverse transform the targets and predictions
    targets_unscaled = scaler_targets.inverse_transform(targets)
    predictions_unscaled = scaler_targets.inverse_transform(predictions)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets_unscaled, predictions_unscaled))
    # rmse =
    mae = mean_absolute_error(targets_unscaled, predictions_unscaled)
    r2 = r2_score(targets_unscaled, predictions_unscaled, force_finite=False)

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
        os.path.join(scaler_targets_dir, "scaler_targets.json"),
        scaler_type="minmax",
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
        feature_r2 = r2_score(
            targets_unscaled[:, i], predictions_unscaled[:, i], force_finite=False
        )
        individual_metrics[f"{feature_name}_rmse"] = feature_rmse
        individual_metrics[f"{feature_name}_mae"] = feature_mae
        individual_metrics[f"{feature_name}_r2"] = feature_r2

    return individual_metrics


def train_test_val_split(directory: str, output_directory: str, tragetfile: str):
    test_intervals = [
        ("2011-08-04", "2011-08-08"),
        ("2017-09-26", "2017-09-29"),
        ("2018-08-13", "2018-08-17"),
        ("2019-08-29", "2019-09-01"),
    ]

    val_years_months = {2014: None, 2017: None}

    # Initialize the DataProcessor
    processor = DataProcessor(directory)

    # Read and merge CSV files from the directory
    # data = processor.read_and_merge_csvs()
    data = pd.read_csv(tragetfile)

    # Create training, validation, and test splits
    train_set, val_set, test_set = processor.create_splits(
        data, test_intervals, val_years_months
    )

    # Save the splits to the local directory
    processor.save_splits(train_set, val_set, test_set, output_directory)


def seed_everything(seed: int):
    """
    Intialize seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            print(file)
            df = pd.read_csv(file_path, low_memory=False)
            df = df.dropna()
            dfs.append(df)

        merged_df = pd.concat(dfs)
        self.column_names = merged_df.columns.tolist()
        print("Total Shape:", merged_df.shape)
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
        self.column_names = data.columns.tolist()
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

    def __init__(
        self, inputfile, targetfile, scaler_dir, is_train=True, get_timestamp=False
    ):
        """
        Initializes the dataset by loading data from a CSV file, scaling it,
        and saving or loading scaler objects. Supports training and testing modes.
        """

        self.inputfile = inputfile
        self.targetfile = targetfile
        self.scaler_dir = scaler_dir
        self.is_train = is_train
        self.get_timestamp = get_timestamp
        self.fl = h5.File(self.inputfile)

        # Read the CSV data from the local directory
        self.dataframe = pd.read_csv(targetfile)

        # # Select input features and target variables
        # self.inputs = self.dataframe.iloc[:, 2:-14]
        self.targets = self.dataframe.iloc[:, 3:]
        self.timestamps = self.dataframe.iloc[:, 1]

        # Initialize scalers and apply them
        if self.is_train:
            # For training data: fit and save scalers
            # self.scaler_inputs = StandardScaler()
            self.scaler_targets = MinMaxScaler()
            # self.inputs = self.scaler_inputs.fit_transform(self.inputs)
            self.targets = self.scaler_targets.fit_transform(self.targets)

            # Save scalers as JSON
            os.makedirs(self.scaler_dir, exist_ok=True)
            # self._save_scaler(self.scaler_inputs, "scaler_inputs.json")
            self._save_scaler(self.scaler_targets, "scaler_targets.json")
        else:
            # For testing data: load and apply existing scalers
            # self.scaler_inputs = self._load_scaler(
            #     "scaler_inputs.json", scaler_type="standard"
            # )
            self.scaler_targets = self._load_scaler(
                "scaler_targets.json", scaler_type="minmax"
            )
            # self.inputs = self.scaler_inputs.transform(self.inputs)
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
        if isinstance(scaler, StandardScaler):
            scaler_dict = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
                "n_features": scaler.n_features_in_,
                "feature_names": scaler.feature_names_in_.tolist(),
            }
        elif isinstance(scaler, MinMaxScaler):
            scaler_dict = {
                "min": scaler.min_.tolist(),
                "scale": scaler.scale_.tolist(),
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                "n_features": scaler.n_features_in_,
                "feature_range": scaler.feature_range,
                "feature_names": scaler.feature_names_in_.tolist(),
            }
        with open(os.path.join(self.scaler_dir, filename), "w") as f:
            json.dump(scaler_dict, f)

    def _load_scaler(self, filename, scaler_type="standard"):
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
        if scaler_type == "standard":
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_dict["mean"])
            scaler.scale_ = np.array(scaler_dict["scale"])
            scaler.var_ = np.array(scaler_dict["var"])
            scaler.n_features_in_ = scaler_dict["n_features"]
            scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])
        elif scaler_type == "minmax":
            scaler = MinMaxScaler(feature_range=tuple(scaler_dict["feature_range"]))
            scaler.min_ = np.array(scaler_dict["min"])
            scaler.scale_ = np.array(scaler_dict["scale"])
            scaler.data_min_ = np.array(scaler_dict["data_min"])
            scaler.data_max_ = np.array(scaler_dict["data_max"])
            scaler.n_features_in_ = scaler_dict["n_features"]
            scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])
        return scaler

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)

    def __del__(self):
        self.fl.close()

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

        ind = self.dataframe.iloc[idx, 2]
        _input = self.fl["latent"][ind, :]
        input_data = torch.tensor(_input, dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.get_timestamp:
            return input_data, target_data, self.timestamps[idx]
        return input_data, target_data


class SHEATH_MLP(nn.Module):
    def __init__(
        self,
        input_dim=21504,
        hidden_dim=2048,
        output_dim=7,
        dropout_rate=0.3,
        init_type="kaiming",
    ) -> None:
        super(SHEATH_MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc5 = nn.Linear(hidden_dim // 4, output_dim)
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
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


def train(config):
    """
    The Main training function which uses wandb to configuration for
    hyperparamete tuning.
    """
    seed_everything(40)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize W&B
    # wandb.init()
    # config = wandb.config

    # Load data
    scaler_dir = "/home/bjha/data/geocloak/formatted_data/sdoembeddings/"
    inputfile = "/home/bjha/data/sdo_latent_dataset_21504.h5"

    print(time.perf_counter())
    train_dataset = SHEATHDataLoader(
        inputfile,
        "/home/bjha/data/geocloak/formatted_data/sdoembeddings/sheath_train_set.csv",
        scaler_dir,
        is_train=True,
    )
    print(time.perf_counter())
    val_dataset = SHEATHDataLoader(
        inputfile,
        "/home/bjha/data/geocloak/formatted_data/sdoembeddings/sheath_val_set.csv",
        scaler_dir,
        is_train=False,
    )
    print(time.perf_counter())
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    print(time.perf_counter())
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    print(time.perf_counter())
    # Define model
    model = SHEATH_MLP()
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_val_rmse = float("inf")
    best_model_state = None

    # Training loop
    # --------------------------------------
    for epoch in range(config["epochs"]):
        print(epoch)
        t1 = time.perf_counter()
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_predictions = []
        for inputs, targets in tq.tqdm(train_loader, desc="Train"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_train_targets.append(targets.cpu().numpy())
            all_train_predictions.append(outputs.detach().cpu().numpy())

        all_train_targets = np.concatenate(all_train_targets, axis=0)
        all_train_predictions = np.concatenate(all_train_predictions, axis=0)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_predictions = []
        with torch.no_grad():
            for inputs, targets in tq.tqdm(val_loader, desc="Valid"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_val_targets.append(targets.cpu().numpy())
                all_val_predictions.append(outputs.cpu().numpy())

        all_val_targets = np.concatenate(all_val_targets, axis=0)
        all_val_predictions = np.concatenate(all_val_predictions, axis=0)

        # Calculate metrics using the saved scaler
        train_rmse, train_mae, train_r2 = calculate_metrics(
            all_train_predictions, all_train_targets, scaler_dir
        )
        val_rmse, val_mae, val_r2 = calculate_metrics(
            all_val_predictions, all_val_targets, scaler_dir
        )

        # Calculate individual metrics for each target feature
        # train_individual_metrics = calculate_individual_metrics(all_train_predictions, all_train_targets, scaler_dir)
        # val_individual_metrics = calculate_individual_metrics(
        #     all_val_predictions, all_val_targets, scaler_dir
        # )

        # Ensure metrics are logged as floats
        train_rmse = float(train_rmse)
        train_mae = float(train_mae)
        train_r2 = float(train_r2)
        val_rmse = float(val_rmse)
        val_mae = float(val_mae)
        val_r2 = float(val_r2)

        # # Log metrics and hyperparameters
        # log_data = {
        #     "epoch": epoch + 1,
        #     "train_loss": running_loss / len(train_loader),
        #     "val_loss": val_loss / len(val_loader),
        #     "train_rmse": train_rmse,
        #     "train_mae": train_mae,
        #     "train_r2": train_r2,
        #     "val_rmse": val_rmse,
        #     "val_mae": val_mae,
        #     "val_r2": val_r2,
        #     "batch_size": wandb.config.batch_size,
        #     "lr": wandb.config.lr,
        #     "weight_decay": wandb.config.weight_decay,
        #     "dropout_rate": wandb.config.dropout_rate,
        # }
        # log_data.update(val_individual_metrics)
        # log_data.update(val_individual_metrics)

        # wandb.log(log_data)

        t2 = time.perf_counter()
        print(
            f'Epoch: {epoch + 1}/{config["epochs"]}, {train_rmse = :.3f},{val_rmse = :.3f}, {t2-t1}',
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()

    best_model_state = model.state_dict()
    # Save the best model checkpoint
    if best_model_state is not None:
        torch.save(
            best_model_state, "../models/sheath_best_model_checkpoint_embed_best.pth"
        )
        print("Best model saved with config to final_sweep.yaml")


if __name__ == "__main__":

    train_test_val_split(
        "/Users/bjha/Data/fdl2024/sheath/alldata/",
        "/home/bjha/data/geocloak/formatted_data/sdoembeddings",
        "/home/bjha/data/geocloak/formatted_data/sdoembeddings/omniweb_back_tracked_ballistic.csv",
    )

    config = {
        "batch_size": 512,
        "dropout_rate": 0.5,
        "epochs": 50,
        "lr": 0.01,
        "weight_decay": 0.01,
    }
    train(config)

    # Load Config file
    # with open("sheath_sweep_config_embed.yml") as f:
    #     sweep_config = yaml.safe_load(f)

    # sweep_id = wandb.sweep(sweep_config, project="sheath_EMBED_sweep")
    # wandb.agent(sweep_id, function=train, count=30)
