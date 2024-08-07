"""
This is main sheath training script for training SHEATH based.
"""

import sys
import os
import warnings
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
import tqdm as tq

warnings.filterwarnings("ignore")

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.models.sheath import (
    DataProcessor,
    SHEATHDataLoader,
    SHEATH_MLP,
    calculate_metrics,
    calculate_individual_metrics,
)


def train_test_val_split(directory: str, output_directory: str):
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
    data = processor.read_and_merge_csvs()

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
    import random, os
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


def train():
    """
    The Main training function which uses wandb to configuration for
    hyperparamete tuning.
    """
    seed_everything(40)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize W&B
    wandb.init()

    # Load data
    directory = "/home/bjha/data/geocloak/formatted_data/sheath_splits"
    scaler_dir = "/home/bjha/data/geocloak/formatted_data/sheath_splits"

    train_dataset = SHEATHDataLoader(
        directory, "sheath_train_set.csv", scaler_dir, is_train=True
    )
    val_dataset = SHEATHDataLoader(
        directory, "sheath_val_set.csv", scaler_dir, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=wandb.config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=wandb.config.batch_size, shuffle=False
    )

    # Define model
    model = SHEATH_MLP(
        input_dim=26,
        hidden_dim=wandb.config.hidden_dim,
        output_dim=14,
        dropout_rate=wandb.config.dropout_rate,
    )
    model.to(device)

    # Log the model architecture and hyperparameters
    wandb.watch(model, log="all")

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay
    )

    # Training loop
    # --------------------------------------
    for epoch in range(wandb.config.epochs):
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
            for inputs, targets in tq.tqdm(val_loader, desc="Validation"):
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
        val_individual_metrics = calculate_individual_metrics(
            all_val_predictions, all_val_targets, scaler_dir
        )

        # Ensure metrics are logged as floats
        train_rmse = float(train_rmse)
        train_mae = float(train_mae)
        train_r2 = float(train_r2)
        val_rmse = float(val_rmse)
        val_mae = float(val_mae)
        val_r2 = float(val_r2)

        # Log metrics and hyperparameters
        log_data = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "batch_size": wandb.config.batch_size,
            "lr": wandb.config.lr,
            "weight_decay": wandb.config.weight_decay,
            "hidden_dim": wandb.config.hidden_dim,
            "dropout_rate": wandb.config.dropout_rate,
        }
        log_data.update(val_individual_metrics)
        # log_data.update(val_individual_metrics)

        wandb.log(log_data)


train_test_split = False
if __name__ == "__main__":
    if train_test_split:
        train_test_val_split(
            "/home/bjha/data/geocloak/formatted_data/sheath_trainv2",
            "/home/bjha/data/geocloak/formatted_data/sheath_splits",
        )
    # Load Config file
    with open("sheath_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="sheath_reg")
    wandb.agent(sweep_id, function=train)
