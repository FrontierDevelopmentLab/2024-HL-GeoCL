import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dataloader import SHEATHDataLoader
from evaluation_metric import calculate_metrics
from model import SHEATH_MLP
from torch.utils.data import DataLoader


def seed_everything(seed: int):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config):
    seed_everything(40)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    directory = "/home/chetrajpandey/data/formatted_data/sheath_splits"
    scaler_dir = "/home/chetrajpandey/data/formatted_data/sheath_splits"

    train_dataset = SHEATHDataLoader(
        directory, "train_set.csv", scaler_dir, is_train=True
    )
    val_dataset = SHEATHDataLoader(directory, "val_set.csv", scaler_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Define model
    model = SHEATH_MLP(
        input_dim=26,
        hidden_dim=config["hidden_dim"],
        output_dim=7,
        dropout_rate=config["dropout_rate"],
    )
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
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_predictions = []
        for inputs, targets in train_loader:
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
            for inputs, targets in val_loader:
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

        # # Calculate individual metrics for each target feature
        # train_individual_metrics = calculate_individual_metrics(
        #     all_train_predictions, all_train_targets, scaler_dir
        # )

        # Ensure metrics are logged as floats
        train_rmse = float(train_rmse)
        train_mae = float(train_mae)
        train_r2 = float(train_r2)
        val_rmse = float(val_rmse)
        val_mae = float(val_mae)
        val_r2 = float(val_r2)

        print(
            f'Epoch {epoch + 1}/{config["epochs"]}, Train RMSE: {train_rmse}, Val RMSE: {val_rmse}'
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()

    # Save the best model checkpoint
    if best_model_state is not None:
        torch.save(best_model_state, "model/best_model_checkpoint_1.pth")
        with open("final_sweep.yaml", "w") as file:
            yaml.dump(config, file)
        print("Best model saved with config to final_sweep.yaml")


if __name__ == "__main__":
    config = {
        "batch_size": 64,
        "hidden_dim": 256,
        "dropout_rate": 0.2,
        "lr": 0.001,
        "weight_decay": 1e-3,
        "epochs": 200,
    }
    train(config)
