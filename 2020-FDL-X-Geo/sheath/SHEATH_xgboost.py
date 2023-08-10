import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from astropy.constants import iau2012 as const
import astropy.units as u
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from pickle import dump, load
import os
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import xgboost as xgb
import optuna

from SHEATH_module import HSENN
from utils.dataloader_torch import Data


# Hyperparameters and other setup info
DATAPATH = "sheath_data/"
param = {'max_depth': 3, 'learning_rate': 1e-3, 'objective': 'reg:squarederror', 'nthread': -1, 'eval_metric': 'rmse'}  # Hyperparameters in xgb notation
num_rounds = 10000
early_stopping_rounds=40


def backtrace_radial(vel):
    """
        Ballistically backtrace a plasma parcel assuming radial flow.
        vel: np.ndarray or scalar in km/s.
    """
    return (const.au.to('km').value)/(vel*3600*24.0)


def get_backtrace_date(vel,sw_date):
    time = backtrace_radial(vel)
    return (sw_date-pd.to_timedelta(time,unit='day')).to_numpy()


def moving_average(array, window_size=3):
    ret = np.cumsum(array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ret[:window_size - 1] = np.nan
    return ret / window_size


# torch.set_default_dtype(torch.float64)  # this is important else it will overflow

if __name__ == "__main__":
    
    timestamps_sun = np.load(f"{DATAPATH}timestamps.npy",allow_pickle=True)
    masks = np.load(f"{DATAPATH}ch_mask.npy")
    ch_net_areas = np.array([np.sum(image) for image in masks])  # Calculate the total area of open-field-line regions    
    
    # Calculate the net flux per measurement per passband
    in_dataset = np.asarray([np.load(v) for v in sorted(glob(f"{DATAPATH}masked*.npy"))]).transpose([1,2,3,0])
    ch_net_fluxes = np.zeros((in_dataset.shape[0],in_dataset.shape[3]))
    for measurement in range(in_dataset.shape[0]):
        for passband in range(in_dataset.shape[3]):
            ch_net_fluxes[measurement, passband] = np.sum(in_dataset[measurement, :, :, passband])
            
    input_data = np.append(ch_net_fluxes, ch_net_areas.reshape((len(ch_net_areas),1)), axis=1)
     
    # Load and process target data
    omni_data = pd.read_hdf(f"{DATAPATH}omni_preprocess.h5",key="omni")
    # omni_data["Proton Temperature, K"] = np.log10(omni_data["Proton Temperature, K"])
    omni_data = omni_data.dropna(axis="rows")
    
    # print(omni_data.shape)
    # for feature in omni_data.columns[1:]:
    #     omni_data[feature] = moving_average(omni_data[feature].values, window_size=90//5)
    # omni_data = omni_data.iloc[::18]
    # print(omni_data.shape)

    aia_dates_omni = get_backtrace_date(omni_data.values[:,5],omni_data.values[:,0])

    # For each "backtraced" index, we now find the nearest AIA/SDO image. This will be our dataset now.

    #Select OMNI data closest to SDO data
    aia_nearest_inds = np.argmin(np.abs(aia_dates_omni[:,None]-timestamps_sun[None,:]),axis=1)
    input_data = np.unique(input_data[aia_nearest_inds], axis=0)
    input_timestamps = np.unique(timestamps_sun[aia_nearest_inds])
    
    omni_nearest_inds = np.argmin(np.abs(aia_dates_omni[:,None]-input_timestamps[None,:]),axis=1)
    omni_nearest_inds = np.unique(omni_nearest_inds)

    omni_data = omni_data[['Date','Field magnitude average, nT', 'BX, nT (GSE, GSM)',
           'BY, nT (GSM)', 'BZ, nT (GSM)', 'Speed, km/s', 'Proton Density, n/cc',
           'Proton Temperature, K']]
    # Downsample omni_data so its indices exactly match input_data's indices
    omni_data = omni_data.iloc[omni_nearest_inds]
    out_feature_names = omni_data.columns.values[1:]
    output_data = omni_data.values[:,1:]
    
    # Save sw variable list
    dump(['Field magnitude average, nT', 'BX, nT (GSE, GSM)',
           'BY, nT (GSM)', 'BZ, nT (GSM)', 'Speed, km/s', 'Proton Density, n/cc',
           'Proton Temperature, K'],open("logs/sw_variables.pickle","wb"))
    
    print(f"Input shape: {input_data.shape}\nOutput shape: {output_data.shape}")


    # Dumb preprocessing: Normalizing everything all at once
    print("Scaling data...")
    scaler_y = MinMaxScaler()
    target_data = output_data
    target_data = scaler_y.fit_transform(output_data)
    # Save scaler y
    dump(scaler_y, open('logs/scaler_y.scaler', 'wb'))
    
    scaler_X = StandardScaler()
    input_data = scaler_X.fit_transform(input_data)
    dump(scaler_X, open('logs/scaler_y.scaler', 'wb'))

    idx = np.arange(input_data.shape[0])
    np.random.seed(2796)
    np.random.shuffle(idx)
    idx = list(idx)
    train_idx = idx[:int(len(idx)*0.85)]
    test_idx = idx[int(len(idx)*0.85):]
    
    print(np.shape(input_data[test_idx]))

    print("Splitting and loading datasets...")
    train_set = xgb.DMatrix(input_data[train_idx], label=target_data[train_idx])
    test_set = xgb.DMatrix(input_data[test_idx], label=target_data[test_idx])
    evallist = [(train_set, 'train'), (test_set, 'eval')]
    
    print("Training XGB Ensemble")
    HSE_gb_model = xgb.train(param, train_set, num_rounds, evallist, early_stopping_rounds=10)
    
    HSE_gb_model.save_model('SHEATH_xgb.model')
    
    predictions = HSE_gb_model.predict(test_set, iteration_range=(0, HSE_gb_model.best_iteration + 1))
    
    print(np.shape(predictions))
    
    fig,axes = plt.subplots(4,2,figsize=(12,10))
    end_timestep = 476
    axes = axes.ravel()
    predictions = scaler_y.inverse_transform(predictions)
    target_data = scaler_y.inverse_transform(target_data)
    for ai in range(len(out_feature_names)):
        axes[ai].plot(omni_data["Date"].iloc[:end_timestep], predictions[:end_timestep, ai], label="Predicted"*(not ai))
        axes[ai].plot(omni_data["Date"].iloc[:end_timestep], target_data[:end_timestep, ai], label="Actual"*(not ai))
        axes[ai].set_title(out_feature_names[ai])
    plt.tight_layout()
    fig.legend()
    plt.savefig("test_timeseries.png")
    
    xgb.plot_importance(HSE_gb_model)
    plt.savefig("xgb_importances")
