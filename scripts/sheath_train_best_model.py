"""
This is main sheath training script for training SHEATH based.
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tq
import yaml
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.models.sheath import (  # noqa: E402
    SHEATH_MLP,
    DataProcessor,
    SHEATHDataLoader,
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
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config, data_dir, scaler_dir, output_checkpoint):
    """
    The Main training function.
    """
    seed_everything(40)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    train_dataset = SHEATHDataLoader(
        data_dir, "sheath_train_set.csv", scaler_dir, is_train=True
    )
    val_dataset = SHEATHDataLoader(
        data_dir, "sheath_val_set.csv", scaler_dir, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Define model
    model = SHEATH_MLP(
        input_dim=26,
        hidden_dim=config["hidden_dim"],
        output_dim=14,
        dropout_rate=config["dropout_rate"],
    )
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Training loop
    for epoch in range(config["epochs"]):
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

    best_model_state = model.state_dict()
    # Save the best model checkpoint
    if best_model_state is not None:
        torch.save(best_model_state, output_checkpoint)
        print(f"Best model saved to {output_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train the SHEATH model.")
    parser.add_argument(
        "--data-dir",
        default="/home/bjha/data/geocloak/formatted_data/sheath_splits",
        help="Directory containing train/val CSV data splits",
    )
    parser.add_argument(
        "--scaler-dir",
        default="/home/bjha/sheath_scalar",
        help="Directory for saving/loading scaler JSON files",
    )
    parser.add_argument(
        "--config",
        default="sheath_best_hyper.yml",
        help="Path to hyperparameter YAML config",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="../models/sheath_best_model_checkpoint_1.pth",
        help="Path to save the trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Run train/test/val split before training",
    )
    parser.add_argument(
        "--raw-data-dir",
        default="/home/bjha/data/geocloak/formatted_data/sheath_trainv2",
        help="Directory with raw CSV files (used with --split)",
    )
    args = parser.parse_args()

    if args.split:
        train_test_val_split(args.raw_data_dir, args.data_dir)

    # Load Config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.data_dir, args.scaler_dir, args.output_checkpoint)


if __name__ == "__main__":
    main()
