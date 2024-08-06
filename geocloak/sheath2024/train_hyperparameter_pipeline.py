import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from dataloader import SHEATHDataLoader
from model import SHEATH_MLP
from evaluation_metric import calculate_metrics, calculate_individual_metrics
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train():
    seed_everything(40)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Initialize W&B
    wandb.init(project='sheath_regression')

    # Load data
    directory = "/home/chetrajpandey/data/formatted_data/sheath_splits"
    scaler_dir = "/home/chetrajpandey/data/formatted_data/sheath_splits"

    train_dataset = SHEATHDataLoader(directory, 'sheath_train_set.csv', scaler_dir, is_train=True)
    val_dataset = SHEATHDataLoader(directory, 'sheath_val_set.csv', scaler_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Define model
    model = SHEATH_MLP(input_dim=26, hidden_dim=wandb.config.hidden_dim, output_dim=14, dropout_rate=wandb.config.dropout_rate)
    model.to(device)

    # Log the model architecture and hyperparameters
    wandb.watch(model, log='all')

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

    # Training loop
    for epoch in range(wandb.config.epochs):
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
        print('Train Targets', all_train_targets)
        print('Train Predictions', all_train_predictions)

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
        train_rmse, train_mae, train_r2 = calculate_metrics(all_train_predictions, all_train_targets, scaler_dir)
        val_rmse, val_mae, val_r2 = calculate_metrics(all_val_predictions, all_val_targets, scaler_dir)

        # Calculate individual metrics for each target feature
        # train_individual_metrics = calculate_individual_metrics(all_train_predictions, all_train_targets, scaler_dir)
        val_individual_metrics = calculate_individual_metrics(all_val_predictions, all_val_targets, scaler_dir)

        # Ensure metrics are logged as floats
        train_rmse = float(train_rmse)
        train_mae = float(train_mae)
        train_r2 = float(train_r2)
        val_rmse = float(val_rmse)
        val_mae = float(val_mae)
        val_r2 = float(val_r2)

        # Log metrics and hyperparameters
        log_data = {
            'epoch': epoch + 1,
            'train_loss': running_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'batch_size': wandb.config.batch_size,
            'lr': wandb.config.lr,
            'weight_decay': wandb.config.weight_decay,
            'hidden_dim': wandb.config.hidden_dim,
            'dropout_rate': wandb.config.dropout_rate
        }
        log_data.update(val_individual_metrics)
        # log_data.update(val_individual_metrics)

        wandb.log(log_data)

if __name__ == "__main__":
    with open('sweep_config.yaml') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project='sth')
    wandb.agent(sweep_id, function=train)
