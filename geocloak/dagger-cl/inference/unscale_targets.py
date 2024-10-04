import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_scaler(scaler_dir, filename):
    """
    Loads a StandardScaler object from a JSON file.

    Args:
        scaler_dir (str): Directory where the scaler JSON files are stored.
        filename (str): Name of the JSON file containing the scaler.

    Returns:
        StandardScaler: The loaded scaler object.
    """
    with open(os.path.join(scaler_dir, filename), "r") as f:
        scaler_dict = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_dict["mean"])
    scaler.scale_ = np.array(scaler_dict["scale"])
    scaler.var_ = np.array(scaler_dict["var"])
    scaler.n_features_in_ = scaler_dict["n_features"]
    scaler.feature_names_in_ = np.array(scaler_dict["feature_names"])
    return scaler


def unscale_predicted_target(
    predicted_target, scaler_dir="/home/jupyter/models/DAGGER_CL"
):
    """
    Unscales the predicted target using the appropriate scalers and prints the unscaled values.

    Args:
        predicted_target (torch.Tensor): The predicted target tensor of shape (1070,).
        scaler_dir (str): Directory where the scaler JSON files are stored.
    """
    assert (
        predicted_target.shape[0] == 1070
    ), "Predicted target must have 1070 dimensions."

    # Load the scalers
    scaler_dbe = load_scaler(scaler_dir, "std_dbe_scaler.json")
    scaler_dbn = load_scaler(scaler_dir, "std_dbn_scaler.json")

    # Split the predicted target
    predicted_dbe = predicted_target[:535].numpy()
    predicted_dbn = predicted_target[535:].numpy()

    # Unscale the predictions
    unscaled_dbe = scaler_dbe.inverse_transform(predicted_dbe.reshape(1, -1)).flatten()
    unscaled_dbn = scaler_dbn.inverse_transform(predicted_dbn.reshape(1, -1)).flatten()

    return unscaled_dbe, unscaled_dbn
