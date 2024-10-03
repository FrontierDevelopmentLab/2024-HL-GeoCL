"""
This script retrieves the best model from a W&B sweep and saves its configuration to a YAML file.
"""

import wandb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml


def get_best_model(runs):
    """
    Find the best model based on a weighted combination of metrics.

    Parameters
    ----------
    runs : list
        List of W&B runs.
    Returns
    -------
    best_run_config: dict
        The configuration of the best run.

    """

    # Extract relevant data from runs
    run_data = []
    for run in runs:
        summary = run.summary._json_dict
        config = run.config
        name = run.name
        metrics = {
            "Bt_mae": summary.get("Bt_mae"),
            "Bt_r2": summary.get("Bt_r2"),
            "Bt_rmse": summary.get("Bt_rmse"),
            "Bx_mae": summary.get("Bx_mae"),
            "Bx_r2": summary.get("Bx_r2"),
            "Bx_rmse": summary.get("Bx_rmse"),
            "By_mae": summary.get("By_mae"),
            "By_r2": summary.get("By_r2"),
            "By_rmse": summary.get("By_rmse"),
            "Bz_mae": summary.get("Bz_mae"),
            "Bz_r2": summary.get("Bz_r2"),
            "Bz_rmse": summary.get("Bz_rmse"),
            "Density_mae": summary.get("Density_mae"),
            "Density_r2": summary.get("Density_r2"),
            "Density_rmse": summary.get("Density_rmse"),
            "Speed_mae": summary.get("Speed_mae"),
            "Speed_r2": summary.get("Speed_r2"),
            "Speed_rmse": summary.get("Speed_rmse"),
            "Temperature_mae": summary.get("Temperature_mae"),
            "Temperature_r2": summary.get("Temperature_r2"),
            "Temperature_rmse": summary.get("Temperature_rmse"),
            "Name": name,
            "Config": config,
        }
        run_data.append(metrics)

    # Convert run data to DataFrame
    runs_df = pd.DataFrame(run_data)

    # Select and weight metrics
    metric_cols = [
        "Bt_mae",
        "Bt_r2",
        "Bt_rmse",
        "Bx_mae",
        "Bx_r2",
        "Bx_rmse",
        "By_mae",
        "By_r2",
        "By_rmse",
        "Bz_mae",
        "Bz_r2",
        "Bz_rmse",
        "Density_mae",
        "Density_r2",
        "Density_rmse",
        "Speed_mae",
        "Speed_r2",
        "Speed_rmse",
        "Temperature_mae",
        "Temperature_r2",
        "Temperature_rmse",
    ]

    metric_weights = np.ones(len(metric_cols))

    # Select metrics from DataFrame
    sel_metrics = runs_df[metric_cols]

    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_metrics = scaler.fit_transform(sel_metrics)

    # Adjust scaling to maximize or minimize metrics
    for i, col in enumerate(metric_cols):
        if "r2" in col.split("_"):
            scaler_metrics[:, i] = scaler_metrics[:, i]  # maximize r2
        else:
            scaler_metrics[:, i] = 1 - scaler_metrics[:, i]  # minimize the rest

    # Weight and combine metrics
    scaler_metrics = scaler_metrics * metric_weights
    scaler_metrics = pd.DataFrame(data=scaler_metrics, columns=metric_cols)
    scaler_metrics["Mean_Metric"] = scaler_metrics.mean(axis=1)
    scaler_metrics["Name"] = runs_df["Name"]
    scaler_metrics["Config"] = runs_df["Config"]
    scaler_metrics = scaler_metrics.sort_values(by="Mean_Metric", ascending=False)

    # Get the config of the best run
    best_run_config = scaler_metrics.iloc[0]["Config"]
    return best_run_config


if __name__ == "__main__":
    # Initialize W&B API
    api = wandb.Api()

    # Project name
    project = "sheath_EMBED_sweep"

    # Get all runs in the project
    runs = api.runs(project)

    best_run_config = get_best_model(runs)

    # Save the config to a YAML file
    with open("sheath_best_hyper_embeded.yml", "w") as file:
        yaml.dump(best_run_config, file)

    print("Best run config saved to 'sheah_best_hyper.yml'")
