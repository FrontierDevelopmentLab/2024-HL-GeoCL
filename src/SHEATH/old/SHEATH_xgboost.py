import os
from glob import glob
from pickle import dump, load

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from astropy.constants import iau2012 as const
from SHEATH_module import HSENN
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data
from tqdm import tqdm
from utils.dataloader_torch import Data

# Hyperparameters and other setup info
DATAPATH = "/home/jupyter/Vishal/fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/sheath_data/"
param = {
    "max_depth": 3,
    "learning_rate": 1e-3,
    "objective": "reg:squarederror",
    "nthread": -1,
    "eval_metric": "rmse",
}  # Hyperparameters in xgb notation
num_rounds = 4000
early_stopping_rounds = 40


def backtrace_radial(vel):
    """
    Ballistically backtrace a plasma parcel assuming radial flow.
    vel: np.ndarray or scalar in km/s.
    """
    return (const.au.to("km").value) / (vel * 3600 * 24.0)


def get_backtrace_date(vel, sw_date):
    time = backtrace_radial(vel)
    return (sw_date - pd.to_timedelta(time, unit="day")).to_numpy()


def moving_average(array, window_size=3):
    ret = np.cumsum(array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ret[: window_size - 1] = np.nan
    return ret / window_size


# torch.set_default_dtype(torch.float64)  # this is important else it will overflow

if __name__ == "__main__":

    print("Loading CH-segmented data")
    timestamps_sun = np.load(f"{DATAPATH}timestamps.npy", allow_pickle=True)
    ch_masks = np.load(f"{DATAPATH}ch_mask.npy")
    ch_net_areas = np.array(
        [np.sum(image) / (image.shape[0] * image.shape[1]) for image in ch_masks]
    )  # Calculate the total area of open-field-line regions

    # Calculate the net flux per measurement per passband
    ch_dataset = np.asarray(
        [np.load(v) for v in sorted(glob(f"{DATAPATH}masked*.npy"))]
    ).transpose([1, 2, 3, 0])
    ch_net_fluxes = np.zeros((ch_dataset.shape[0], ch_dataset.shape[3]))
    for measurement in range(ch_dataset.shape[0]):
        for passband in range(ch_dataset.shape[3]):
            ch_net_fluxes[measurement, passband] = np.sum(
                ch_dataset[measurement, :, :, passband]
            )
    ch_data = np.append(
        ch_net_fluxes, ch_net_areas.reshape((len(ch_net_areas), 1)), axis=1
    )

    print("Loading AR-segmented data")
    ar_masks = np.load(f"{DATAPATH}ar_mask.npy")
    ar_net_areas = np.array(
        [np.sum(image) / (image.shape[0] * image.shape[1]) for image in ar_masks]
    )  # Calculate the total area of open-field-line regions

    # Calculate the net flux per measurement per passband
    ar_dataset = np.asarray(
        [np.load(v) for v in sorted(glob(f"{DATAPATH}ar_masked*.npy"))]
    ).transpose([1, 2, 3, 0])
    ar_net_fluxes = np.zeros((ar_dataset.shape[0], ar_dataset.shape[3]))
    for measurement in range(ar_dataset.shape[0]):
        for passband in range(ar_dataset.shape[3]):
            ar_net_fluxes[measurement, passband] = np.sum(
                ar_dataset[measurement, :, :, passband]
            )

    ar_data = np.append(
        ar_net_fluxes, ar_net_areas.reshape((len(ar_net_areas), 1)), axis=1
    )
    input_data = np.append(ch_data, ar_data, axis=1)

    # Load and process target data
    omni_data = pd.read_hdf(f"{DATAPATH}omni_preprocess.h5", key="omni")
    # a = pd.read_hdf("../data_local/omni/sw_data.h5", key="2010")

    # omni_data["Proton Temperature, K"] = np.log10(omni_data["Proton Temperature, K"])
    omni_data = omni_data.dropna(axis="rows")

    print(omni_data.shape)
    # for feature in omni_data.columns[1:]:
    #     omni_data[feature] = moving_average(omni_data[feature].values, window_size=90//5)
    # omni_data = omni_data.iloc[::18]
    print(omni_data.shape)

    aia_dates_omni = get_backtrace_date(omni_data.values[:, 5], omni_data.values[:, 0])

    # For each "backtraced" index, we now find the nearest AIA/SDO image. This will be our dataset now.

    # Select OMNI data closest to SDO data
    aia_nearest_inds = np.argmin(
        np.abs(aia_dates_omni[:, None] - timestamps_sun[None, :]), axis=1
    )
    input_data = np.unique(input_data[aia_nearest_inds], axis=0)
    input_timestamps = np.unique(timestamps_sun[aia_nearest_inds])

    omni_nearest_inds = np.argmin(
        np.abs(aia_dates_omni[:, None] - input_timestamps[None, :]), axis=1
    )
    omni_nearest_inds = np.unique(omni_nearest_inds)

    omni_data = omni_data[
        [
            "Date",
            "BX, nT (GSE, GSM)",
            "BY, nT (GSM)",
            "BZ, nT (GSM)",
            "Speed, km/s",
            "Proton Density, n/cc",
            "Proton Temperature, K",
        ]
    ]
    # omni_data = omni_data[['Date','BZ, nT (GSM)']]
    # Downsample omni_data so its indices exactly match input_data's indices
    omni_data = omni_data.iloc[omni_nearest_inds]
    out_feature_names = omni_data.columns.values[1:]
    output_data = omni_data.values[:, 1:]

    # Save sw variable list
    dump(
        [
            "BX, nT (GSE, GSM)",
            "BY, nT (GSM)",
            "BZ, nT (GSM)",
            "Speed, km/s",
            "Proton Density, n/cc",
            "Proton Temperature, K",
        ],
        open("logs/sw_variables.pickle", "wb"),
    )

    print(f"Input shape: {input_data.shape}\nOutput shape: {output_data.shape}")

    # Dumb preprocessing: Normalizing everything all at once
    print("Scaling data...")
    scaler_y = MinMaxScaler()
    target_data = scaler_y.fit_transform(output_data)
    # Save scaler y
    dump(scaler_y, open("logs/scaler_y.scaler", "wb"))

    scaler_X = StandardScaler()
    input_data = scaler_X.fit_transform(input_data)
    dump(scaler_X, open("logs/scaler_X.scaler", "wb"))

    idx = np.arange(input_data.shape[0])
    np.random.seed(2796)
    np.random.shuffle(idx)
    idx = list(idx)
    train_idx = idx[: int(len(idx) * 0.85)]
    test_idx = idx[int(len(idx) * 0.85) :]
    test_timestamps = input_timestamps[test_idx]

    print("Splitting and loading datasets...")
    train_set = xgb.DMatrix(input_data[train_idx], label=target_data[train_idx])
    test_set = xgb.DMatrix(input_data[test_idx], label=target_data[test_idx])
    evallist = [(train_set, "train"), (test_set, "eval")]

    print("Training XGB Ensemble")
    HSE_gb_model = xgb.train(
        param, train_set, num_rounds, evallist, early_stopping_rounds=10
    )

    HSE_gb_model.save_model("SHEATH_xgb.model")

    predictions = HSE_gb_model.predict(
        test_set, iteration_range=(0, HSE_gb_model.best_iteration + 1)
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    end_timestep = 400
    axes = axes.ravel()

    # Uncomment the following line if there is only one feature in your target dataset.
    # predictions = predictions.reshape((len(predictions),1))

    predictions = scaler_y.inverse_transform(predictions)
    target_data = scaler_y.inverse_transform(target_data)
    for ai in range(len(out_feature_names)):
        axes[ai].plot(
            omni_data["Date"].iloc[:end_timestep],
            predictions[:end_timestep, ai],
            label="Predicted" * (not ai),
        )
        axes[ai].plot(
            omni_data["Date"].iloc[:end_timestep],
            target_data[:end_timestep, ai],
            label="Actual" * (not ai),
        )
        axes[ai].tick_params(axis="x", labelrotation=45)
        axes[ai].set_title(
            out_feature_names[ai]
            + f" MSE: {mean_squared_error(predictions[:end_timestep, ai], target_data[:end_timestep, ai])}"
        )
    plt.tight_layout()
    fig.legend()
    plt.savefig("test_timeseries.png")

    xgb.plot_importance(HSE_gb_model)
    plt.savefig("xgb_importances")

    predictions = pd.DataFrame(
        predictions, columns=["bx", "by", "bz", "vx", "density", "temperature"]
    )
    predictions["Time"] = test_timestamps
    predictions["vz"] = np.zeros(len(predictions))  # Assume velocity is purely radial
    predictions["vy"] = np.zeros(len(predictions))
    predictions["xgse"] = np.zeros(len(predictions))
    predictions["ygse"] = np.zeros(len(predictions))
    predictions["zgse"] = np.zeros(len(predictions))
    predictions["clock_angle"] = np.arctan(predictions["by"] / predictions["bz"])
    predictions["psw"] = (
        (predictions["vx"] * 1000) ** 2 * predictions["density"] * 1e6 * 1.6726e-27
    )  # Ram pressure in radial, Hydrogen solar wind
    predictions.sort_values(by="Time", inplace=True)
    predictions.set_index("Time", inplace=True, drop=True)
    predictions.index = predictions.index.floor(freq="min")

    # Ensure columns are in the right order
    predictions = predictions[
        [
            "bx",
            "by",
            "bz",
            "vx",
            "vy",
            "vz",
            "density",
            "psw",
            "temperature",
            "xgse",
            "ygse",
            "zgse",
            "clock_angle",
        ]
    ]

    predictions.to_hdf("../data_local/omni/sheath_sw_data.h5", key="2010")

    print("Inference complete!")
