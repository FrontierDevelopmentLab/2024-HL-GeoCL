import os
from glob import glob
from pickle import dump, load

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.constants import iau2012 as const
from SHEATH_module import HSENN
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data
from tqdm import tqdm
from utils.dataloader_torch import Data

# Hyperparameters and other setup info
DATAPATH = "sheath_data/"
batch_size = 100
lr = 1e-3
num_epochs = 250
loss_fn = nn.MSELoss()
if not os.path.isdir("logs/"):
    os.makedirs("logs/")


def backtrace_radial(vel):
    """
    Ballistically backtrace a plasma parcel assuming radial flow.
    vel: np.ndarray or scalar in km/s.
    """
    return (const.au.to("km").value) / (vel * 3600 * 24.0)


def get_backtrace_date(vel, sw_date):
    time = backtrace_radial(vel)
    return (sw_date - pd.to_timedelta(time, unit="day")).to_numpy()


# torch.set_default_dtype(torch.float64)  # this is important else it will overflow

if __name__ == "__main__":

    timestamps_sun = np.load(f"{DATAPATH}timestamps.npy", allow_pickle=True)
    in_dataset = np.asarray(
        [np.load(v) for v in sorted(glob(f"{DATAPATH}masked*.npy"))]
    ).transpose([1, 2, 3, 0])

    omni_data = pd.read_hdf(f"{DATAPATH}omni_preprocess.h5", key="omni")
    # omni_data["Proton Temperature, K"] = np.log10(omni_data["Proton Temperature, K"])
    omni_data = omni_data.dropna(axis="rows")

    aia_dates_omni = get_backtrace_date(omni_data.values[:, 5], omni_data.values[:, 0])

    omni_data = omni_data[
        [
            "Date",
            "Field magnitude average, nT",
            "BX, nT (GSE, GSM)",
            "BY, nT (GSM)",
            "BZ, nT (GSM)",
            "Speed, km/s",
            "Proton Density, n/cc",
            "Proton Temperature, K",
        ]
    ]
    output_data = omni_data.values[:, 1:]
    # Save sw variable list
    dump(
        [
            "Field magnitude average, nT",
            "BX, nT (GSE, GSM)",
            "BY, nT (GSM)",
            "BZ, nT (GSM)",
            "Speed, km/s",
            "Proton Density, n/cc",
            "Proton Temperature, K",
        ],
        open("logs/sw_variables.pickle", "wb"),
    )

    # For each "backtraced" index, we now find the nearest AIA/SDO image. This will be our dataset now.

    # Select OMNI data closest to SDO data
    aia_nearest_inds = np.argmin(
        np.abs(aia_dates_omni[:, None] - timestamps_sun[None, :]), axis=1
    )
    input_data = in_dataset[aia_nearest_inds]
    input_timestamps = timestamps_sun[aia_nearest_inds]

    print(f"Input shape: {input_data.shape}\nOutput shape: {output_data.shape}")

    # Dumb preprocessing: Normalizing everything all at once
    print("Scaling data...")
    scaler_y = MinMaxScaler()
    target_data = scaler_y.fit_transform(output_data)
    # Save scaler y
    dump(scaler_y, open("logs/scaler_y.scaler", "wb"))

    scaler_X = StandardScaler()
    input_data = scaler_X.fit_transform(
        input_data.reshape(-1, input_data.shape[-1])
    ).reshape(input_data.shape)
    dump(scaler_X, open("logs/scaler_X.scaler", "wb"))

    idx = np.arange(input_data.shape[0])
    np.random.seed(2796)
    np.random.shuffle(idx)
    idx = list(idx)
    train_idx = idx[: int(len(idx) * 0.85)]
    test_idx = idx[int(len(idx) * 0.85) :]

    print("Splitting and loading datasets...")
    train_set = Data(input_data[train_idx], target_data[train_idx])
    test_set = Data(input_data[test_idx], target_data[test_idx])

    training_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        test_set, batch_size=test_set.features.shape[0], shuffle=True
    )

    # This defines our model
    HSE_model = HSENN(
        n_passbands=train_set.n_passbands,
        height=train_set.height,
        width=train_set.width,
        n_out=train_set.nout,
    )
    HSE_model = HSE_model.to(train_set.device).float()
    model_hyperparams = {
        "n_passbands": train_set.n_passbands,
        "height": train_set.height,
        "width": train_set.width,
        "n_out": train_set.nout,
    }
    dump(model_hyperparams, open("logs/model_hyperparams.scaler", "wb"))
    # print(HSE_model)
    optimizer = torch.optim.Adam(HSE_model.parameters(), lr=lr)

    # Train the model!
    training_stats = []
    testing_stats = []
    for i in tqdm(np.arange(num_epochs), desc="Training model"):
        ce_loss = []
        for features, target in training_loader:
            optimizer.zero_grad()

            predictions = HSE_model(features)

            loss = loss_fn(predictions, target)

            loss.backward()
            optimizer.step()

            ce_loss.append(loss.detach().cpu().numpy())
        training_stats.append(np.mean(ce_loss))
        with torch.no_grad():
            ce_loss = []
            for features, target in test_loader:

                predictions = HSE_model(features)

                loss = loss_fn(predictions, target)
                ce_loss.append(loss.detach().cpu().numpy())

            testing_stats.append(np.mean(ce_loss))
        if i % 100 == 0:
            print(f"Training loss: {training_stats[-1]}")
            print(f"Testing loss: {testing_stats[-1]}")

    plt.plot(training_stats, "r", label="Training")
    plt.plot(testing_stats, label="Testing")
    plt.legend()
    plt.savefig("train_test_loss.png")

    torch.save(HSE_model.state_dict(), "logs/SHEATH.ckpt")

    with torch.no_grad():
        preds = []
        for features, target in test_loader:

            predictions = HSE_model(features)
            preds.append(predictions.detach().cpu().numpy())

    _, ytest_targ = test_set[:]
    ytest_targ = ytest_targ.detach().cpu().numpy()
    preds = np.concatenate(preds, axis=0)

    plt.figure()
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ai in range(train_set.nout):
        axes[ai].scatter(ytest_targ[:, ai], preds[:, ai])
        axes[ai].set_xlabel("Target values (normalized)")
        axes[ai].set_ylabel("Predicted values (normalized)")
    plt.tight_layout()
    plt.savefig("target_vs_predicted.png")
