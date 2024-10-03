import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_scaler_from_json(filename):
    """
    Load a StandardScaler object from a JSON file.

    Args:
    - filename (str): Path to the JSON file containing the scaler.

    Returns:
    - scaler (StandardScaler): The loaded scaler object.
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

    Args:
    - predictions (np.ndarray): Model predictions (scaled).
    - targets (np.ndarray): True targets (scaled).
    - scaler_targets_dir (str): Directory path where scaler JSON is saved.

    Returns:
    - metrics (dict): A dictionary containing RMSE, MAE, and R^2 scores.
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

    Args:
    - predictions (np.ndarray): Model predictions (scaled).
    - targets (np.ndarray): True targets (scaled).
    - scaler_targets_dir (str): Directory path where scaler JSON is saved.

    Returns:
    - metrics (dict): A dictionary containing individual RMSE, MAE, and R^2 scores for each feature.
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
