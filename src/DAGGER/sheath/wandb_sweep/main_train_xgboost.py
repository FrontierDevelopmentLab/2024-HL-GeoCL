import wandb
from wandb.xgboost import WandbCallback
import numpy as np
import yaml
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
from scipy.stats import pearsonr

DATAPATH = "/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/"
num_rounds = 20000
early_stopping_rounds=40

# hyperparameter_best = {'verbosity': 1, 
#                       'objective': 'reg:squarederror',
#                       'nthread': -1,'eval_metric': 'rmse',
#                       'eta':config.eta, 'gamma':config.gamma, 
#                       'max_depth':config.max_depth, 'subsample':config.subsample, 
#                       'reg_alpha':config.reg_alpha}
# hyperparameter_defaults = hyperparameter_best
with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
wandb.init(project="sheath_xgboost" , config=config)
# config = wandb.config

def train(): 
    # Load AIA data
    config = wandb.config
    print("###############==============\n New day, new run\n##############=============")
    print(config)
    # Load AIA data
    
    list_aia = sorted(glob(f"{DATAPATH}sheath_aia_data/aia_subsamp_masked_summed_*.npy"))
    aia_featureset =  np.concatenate([np.load(a).T for a in list_aia],axis=0)    
    
    # Load AIA times
    list_aia = sorted(glob(f"{DATAPATH}sheath_aia_data/AIA193_times_*.npy"))
    aia_times = np.concatenate([np.load(a) for a in list_aia])
        
    # Load OMNI data and dates
    omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_1hour_full.h5"
    omni_data = pd.read_hdf(omni_path)
    out_feature_names = ['BX, nT (GSE, GSM)', 'BY, nT (GSM)', 'BZ, nT (GSM)', 'Speed, km/s', 
                           'Proton Density, n/cc', 'Proton Temperature, K']
    omni_data = omni_data[out_feature_names]
    # omni_inds_path = sorted(glob(f"{DATAPATH}logs/closest_omni_aia_indices_*.npy"))
    # omni_inds_closest_to_aia = np.concatenate([np.load(a) for a in omni_inds_path])
    
    # omni_dates_path = sorted(glob(f"{DATAPATH}logs/closest_omni_aia_dates_*.npy"))
    # omni_dates_closest_to_aia = np.concatenate([np.load(a) for a in omni_dates_path])
    omni_data_closest_aia = omni_data#.iloc[omni_inds_closest_to_aia]
    #==== Now we have OMNI data corresponding to AIA data. We need to split this data into different datasets.
    #Load OMNI dates and indices
    omni_inds = np.load(f"{DATAPATH}logs/sheath_train_test_val_split.npz", allow_pickle=True)
    train_inds = omni_inds['train']
    val_inds = omni_inds['val']
    test_inds = omni_inds['test']
    storm_2017 = omni_inds['storm_2017']
    storm_2015 = omni_inds['storm_2015']
    storm_2011 = omni_inds['storm_2011']
    
    
    # Dumb preprocessing: Normalizing everything all at once
    print("Scaling data...")
    scaler_y = MinMaxScaler()
    target_data = scaler_y.fit_transform(omni_data_closest_aia)
    # Save scaler y
    # dump(scaler_y, open('../logs/scaler_y.scaler', 'wb'))
    
    scaler_X = StandardScaler()
    input_data = scaler_X.fit_transform(aia_featureset)
    # dump(scaler_X, open('../logs/scaler_X.scaler', 'wb'))
    
    # Save sw variable list
    
    # dump(out_feature_names,open("../logs/sw_variables.pickle","wb"))
    
    print(f"Input shape: {input_data.shape}\nOutput shape: {target_data.shape}")

    # omni_data = omni_data[['Date','BZ, nT (GSM)']]
    # Downsample omni_data so its indices exactly match input_data's indices
    print("Splitting the data into train-test-val-storms")
    x_train,y_train = input_data[train_inds], target_data[train_inds]
    x_val,y_val = input_data[val_inds], target_data[val_inds]
    x_test,y_test = input_data[test_inds], target_data[test_inds]
    x_2017,y_2017 = input_data[storm_2017], target_data[storm_2017]
    x_2015,y_2015 = input_data[storm_2015], target_data[storm_2015]
    x_2011,y_2011 = input_data[storm_2011], target_data[storm_2011]    
    
    print("Splitting and loading datasets...")
    train_set = xgb.DMatrix(x_train, label=y_train)
    test_set = xgb.DMatrix(x_test, label=y_test)
    val_set = xgb.DMatrix(x_val, label=y_val)
    set_2017 = xgb.DMatrix(x_2017, label=y_2017)
    set_2015 = xgb.DMatrix(x_2015, label=y_2015)
    set_2011 = xgb.DMatrix(x_2011, label=y_2011)
    
    evallist = [(train_set, 'train'), (test_set, 'test'), (val_set, 'val'),
               (set_2017, '2017'), (set_2015, '2015'), (set_2011, '2011')]
    
    bst_params = {'verbosity': 1, 
                  'objective': config.loss,
                  'nthread': -1,
                  'eta':config.eta,
                  'max_depth':config.max_depth, 'subsample':config.subsample, 
                  'reg_alpha':config.reg_alpha}


    # Initialize the XGBoostClassifier with the WandbCallback
    # Train the model
    HSE_gb_model = xgb.train(bst_params, train_set, num_rounds, evals = evallist, early_stopping_rounds=config.early_stopping_rounds)

    # Log booster metrics

    # Get train and validation predictions
    preds = [HSE_gb_model.predict(ds[0], iteration_range=(0, HSE_gb_model.best_iteration + 1)) for ds in evallist]
    targs = [y_train,y_test,y_val,y_2017,y_2015,y_2011]
    val_mse = mean_squared_error(preds[2], targs[2])
    train_mse = mean_squared_error(preds[0], targs[0])
    test_mse = mean_squared_error(preds[1], targs[1])
    hse_mse = mean_squared_error(preds[3], targs[3])

    val_correl = np.min([pearsonr(preds[2][:,i], targs[2][:,i])[0] for i in range(preds[0].shape[-1])])
    train_correl = np.min([pearsonr(preds[0][:,i], targs[0][:,i])[0] for i in range(preds[0].shape[-1])])
    test_correl = np.min([pearsonr(preds[1][:,i], targs[1][:,i])[0] for i in range(preds[0].shape[-1])])
    hse_correl = np.min([pearsonr(preds[3][:,i], targs[3][:,i])[0] for i in range(preds[0].shape[-1])])

    # Log metrics
    wandb.log({'val_loss': val_mse})
    wandb.log({'val_correl': val_correl})
    wandb.log({'train_correl': train_correl})
    wandb.log({'test_correl': test_correl})
    wandb.log({'hse_correl': hse_correl})
    wandb.log({'train_mse': train_mse})
    wandb.log({'test_mse': test_mse})
    wandb.log({'hse_mse': hse_mse})
    # , 'train_mse': train_mse, 'test_mse': test_mse, 'hse_mse': hse_mse,
               # 'val_correl': val_correl, 'train_correl': train_correl, 'test_correl':test_correl, 'hse_correl': hse_correl})
        
if __name__ == '__main__':
    train()