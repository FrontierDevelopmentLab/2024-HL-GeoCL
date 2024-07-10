import os
import time
from glob import glob
from pathlib import Path
from pickle import dump, load

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from astropy.constants import iau2012 as const
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data
from tqdm import tqdm

# Hyperparameters and other setup info
DATAPATH = "/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/"
param = {
    "verbosity": 1,
    "objective": "reg:pseudohubererror",
    "nthread": -1,
    "eta": 2e-3,
    "max_depth": 3052,
    "subsample": 0.2,
    "reg_alpha": 0.005,
}

# param = {'max_depth': 12, 'learning_rate': 1e-3, 'objective': 'reg:squarederror', 'nthread': -1, 'eval_metric': 'rmse'}  # Hyperparameters in xgb notation
num_rounds = 15000
early_stopping_rounds = 30


# torch.set_default_dtype(torch.float64)  # this is important else it will overflow

if __name__ == "__main__":

    while True:
        # Be trapped here till the mask file is actually being generated.
        my_file = Path(
            f"/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/sheath_aia_data/aia_subsamp_masked_summed_2012.npy"
        )
        if my_file.is_file():
            break
        else:
            print("Yet to have all the files processed.....")
            time.sleep(600)
    print("All files have been processed and are ready... Starting to fit XGBoost.....")

    # Load AIA data

    list_aia = sorted(
        glob(f"{DATAPATH}sheath_aia_data/aia_subsamp_masked_summed_*.npy")
    )
    aia_featureset = np.concatenate([np.load(a).T for a in list_aia], axis=0)

    # Load AIA times
    list_aia = sorted(glob(f"{DATAPATH}sheath_aia_data/AIA193_times_*.npy"))
    aia_times = np.concatenate([np.load(a) for a in list_aia])

    # Load OMNI data and dates
    omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_1hour_full.h5"
    omni_data = pd.read_hdf(omni_path)
    out_feature_names = [
        "BX, nT (GSE, GSM)",
        "BY, nT (GSM)",
        "BZ, nT (GSM)",
        "Speed, km/s",
        "Proton Density, n/cc",
        "Proton Temperature, K",
    ]
    omni_data = omni_data[out_feature_names]
    # omni_inds_path = sorted(glob(f"{DATAPATH}logs/closest_omni_aia_indices_*.npy"))
    # omni_inds_closest_to_aia = np.concatenate([np.load(a) for a in omni_inds_path])

    # omni_dates_path = sorted(glob(f"{DATAPATH}logs/closest_omni_aia_dates_*.npy"))
    # omni_dates_closest_to_aia = np.concatenate([np.load(a) for a in omni_dates_path])
    omni_data_closest_aia = omni_data  # .iloc[omni_inds_closest_to_aia]
    # ==== Now we have OMNI data corresponding to AIA data. We need to split this data into different datasets.
    # Load OMNI dates and indices
    omni_inds = np.load(
        f"{DATAPATH}logs/sheath_train_test_val_split.npz", allow_pickle=True
    )
    train_inds = omni_inds["train"]
    val_inds = omni_inds["val"]
    test_inds = omni_inds["test"]
    storm_2017 = omni_inds["storm_2017"]
    storm_2015 = omni_inds["storm_2015"]
    storm_2011 = omni_inds["storm_2011"]

    # Dumb preprocessing: Normalizing everything all at once
    print("Scaling data...")
    scaler_y = MinMaxScaler()
    target_data = scaler_y.fit_transform(omni_data_closest_aia)
    # Save scaler y
    dump(scaler_y, open("logs/scaler_y.scaler", "wb"))

    scaler_X = StandardScaler()
    input_data = scaler_X.fit_transform(aia_featureset)
    dump(scaler_X, open("logs/scaler_X.scaler", "wb"))

    # Save sw variable list

    dump(out_feature_names, open("logs/sw_variables.pickle", "wb"))

    print(f"Input shape: {input_data.shape}\nOutput shape: {target_data.shape}")

    # omni_data = omni_data[['Date','BZ, nT (GSM)']]
    # Downsample omni_data so its indices exactly match input_data's indices
    print("Splitting the data into train-test-val-storms")
    x_train, y_train = input_data[train_inds], target_data[train_inds]
    x_val, y_val = input_data[val_inds], target_data[val_inds]
    x_test, y_test = input_data[test_inds], target_data[test_inds]
    x_2017, y_2017 = input_data[storm_2017], target_data[storm_2017]
    x_2015, y_2015 = input_data[storm_2015], target_data[storm_2015]
    x_2011, y_2011 = input_data[storm_2011], target_data[storm_2011]

    print("Splitting and loading datasets...")
    train_set = xgb.DMatrix(x_train, label=y_train)
    test_set = xgb.DMatrix(x_test, label=y_test)
    val_set = xgb.DMatrix(x_val, label=y_val)
    set_2017 = xgb.DMatrix(x_2017, label=y_2017)
    set_2015 = xgb.DMatrix(x_2015, label=y_2015)
    set_2011 = xgb.DMatrix(x_2011, label=y_2011)

    evallist = [
        (train_set, "train"),
        (test_set, "test"),
        (val_set, "val"),
        (set_2017, "2017"),
        (set_2015, "2015"),
        (set_2011, "2011"),
    ]

    print("Training XGB Ensemble")
    HSE_gb_model = xgb.train(
        param, train_set, num_rounds, evallist, early_stopping_rounds=10
    )

    HSE_gb_model.save_model("logs/SHEATH_xgb.model")

    predictions = [
        HSE_gb_model.predict(
            ds[0], iteration_range=(0, HSE_gb_model.best_iteration + 1)
        )
        for ds in evallist
    ]
    np.savez(
        "logs/model_predictions.npz",
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_2017=y_2017,
        y_2015=y_2015,
        y_2011=y_2011,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        x_2017=x_2017,
        x_2015=x_2015,
        x_2011=x_2011,
        predictions_train=predictions[0],
        predictions_test=predictions[1],
        predictions_val=predictions[2],
        predictions_2017=predictions[3],
        predictions_2015=predictions[4],
        predictions_2011=predictions[5],
    )

    labels = [ds[1] for ds in evallist]
    print("Predicting and computing statistics, saving....")
    for preds, targs, label in zip(
        predictions, [y_train, y_test, y_val, y_2017, y_2015, y_2011], labels
    ):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.ravel()
        for ai in range(len(out_feature_names)):
            axes[ai].scatter(targs, preds, label=out_feature_names[ai])
            axes[ai].set_xlabel("Target")
            axes[ai].set_xlabel("Predictions")
            mse = mean_squared_error(preds[:, ai], targs[:, ai])
            correl = pearsonr(preds[:, ai], targs[:, ai])[0]
            axes[ai].set_title(label + f" MSE: {mse:.2f}, Correlation: {correl:.2f}")
        plt.tight_layout()
        fig.legend()
        plt.savefig(f"{label}_statistics.png")

    # predictions = scaler_y.inverse_transform(predictions)
    #     target_data = scaler_y.inverse_transform(target_data)

    #     xgb.plot_importance(HSE_gb_model)
    #     plt.savefig("xgb_importances")

    #     predictions = pd.DataFrame(predictions, columns=['bx',
    #            'by', 'bz', 'vx', 'density', 'temperature'])
    #     predictions["Time"] = test_timestamps
    #     predictions['vz'] = np.zeros(len(predictions))  # Assume velocity is purely radial
    #     predictions['vy'] = np.zeros(len(predictions))
    #     predictions['xgse'] = np.zeros(len(predictions))
    #     predictions['ygse'] = np.zeros(len(predictions))
    #     predictions['zgse'] = np.zeros(len(predictions))
    #     predictions['clock_angle'] = np.arctan(predictions['by']/predictions['bz'])
    #     predictions['psw'] = (predictions['vx']*1000)**2 * predictions['density'] * 1e6 * 1.6726e-27  # Ram pressure in radial, Hydrogen solar wind
    #     predictions.sort_values(by='Time', inplace=True)
    #     predictions.set_index("Time", inplace=True, drop=True)
    #     predictions.index = predictions.index.floor(freq="min")

    #     # Ensure columns are in the right order
    #     predictions = predictions[["bx", "by", "bz", "vx", "vy", "vz", "density", "psw", "temperature", "xgse", "ygse", "zgse", "clock_angle"]]

    #     predictions.to_hdf("../data_local/omni/sheath_sw_data.h5", key="2010")

    print("Inference complete!")
