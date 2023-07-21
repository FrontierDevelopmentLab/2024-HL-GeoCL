import os.path
import os
import pickle
import pandas as pd
import h5py
import numpy as np
import pytorch_lightning as pl
import torch.optim
import wandb
from astropy.time import Time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.geoeffectivenet import *
from models.spherical_harmonics import SphericalHarmonics
from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data,get_wiemer_data,get_iaga_data_as_list, get_iaga_max_stations, get_input_data
from utils.splitter import generate_indices
from dataloader import OMNIDataset, ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset, InputDataset
from tqdm.contrib.concurrent import process_map
from utils.helpers import dipole_tilt
import tqdm
import pickle

# Preprocessing parameters
past_omni_length = 120
lag = 30
yearlist = list(np.arange(2010,2018+1).astype(int))
targets = ["dbe_nez", "dbn_nez"]
future_length = 1
nmax = 20
num_workers = 32
output_folder = './processed_data_all_years'


class PreprocessData():
    def __init__(self,
        supermag_data,
        omni_data,
        idx,
        f107_dataset,
        targets="dbn_nez",
        past_omni_length=120,
        past_supermag_length=10,
        future_length=10,
        lag=0,
        zero_omni=False,
        zero_supermag=False,
        scaler=None,
        training_batch=True,
        nmax=20,
        inference=False,
        num_workers = 4,
        category = 'train'):
        
        self.supermag_data = supermag_data.data
        
        self.supermag_features = supermag_data.features

        # Generate the slices correspondong to each bucket
        self.sg_indices = idx
        
        #Now use new_inds to index the array elements. 
        #Size of data should now be [N_buckets,N_elements_in_bucket,...]

        self.dates = supermag_data.dates
        # del new_inds
        #shape (n_buckets,n_elements_in_bucket)

        self.target_idx = []
        for target in targets:
            self.target_idx.append(np.where(self.supermag_features == target)[0][0])

        self.omni = omni_data.data.values
        #This shape is (n_total,n_omni)

        print("extracting f107")
        self.f107path = f107_dataset
        f107_data = np.load(f107_dataset)
        
        # Vectorized operation: pd datetime needs 1D array, and give unit as 's'
        # tmp_dates = pd.to_datetime(self.dates.reshape(-1),unit='s').to_numpy().reshape(list(self.dates.shape))
        # #Find the best matching f10.7 index along 3rd dimension
        # match = np.argmin(np.abs(np.expand_dims(tmp_dates,axis=-1)-f107_data["dates"].reshape([1,1,-1])),axis=-1)
        # del tmp_dates
        self.f107 = [f107_data["f107"],f107_data["dates"]]
        # of shape (n_buckets, n_points_per_buckets)

        # add dipole
        
        omni_columns = np.array(omni_data.data.columns.tolist() + ["dipole", "f107"])

        self.omni_features = omni_columns

        self.targets = targets
        self.window_length = past_omni_length+lag-1
        self.past_omni_length = past_omni_length
        self.past_supermag_length = past_supermag_length
        self.future_length = future_length
        self.lag = lag

        if scaler is not None:
            print("using existing scaler")
            self.scaler = scaler
            if inference:
                omni_mean, omni_std = scaler["omni"]
                self.omni = (self.omni-omni_mean[:-2])/omni_std[:-2]

                target_mean, target_std = scaler["supermag"]
                self.supermag_data[...,self.target_idx] = (self.supermag_data[...,self.target_idx]-target_mean)/target_std
        else:
            self.compute_scaler(f107_data)
        self._nbasis = nmax
       
        #self.features_list = process_map(self.process_data, [i for i in range(len(self.sg_indices))], max_workers = num_workers, chunksize = 256)
        
    def compute_features(self):
        self.sg_indices_list = []
        for i in tqdm.trange(len(self.sg_indices), desc="Inital 'bucketizing'"):
            sg_ind = self.sg_indices[i]
            po = self.omni[sg_ind[0]:sg_ind[0]+self.past_omni_length,...]
            past_supermag = self.supermag_data[sg_ind[0],...][None,:]
            past_dates = self.dates[sg_ind[0]:sg_ind[0]+self.past_omni_length]
            dp = (dipole_tilt(self.dates[sg_ind[0]:sg_ind[0]+self.past_omni_length])-self.scaler["omni"][0][-2])/(self.scaler["omni"][0][-2])
            tmp_dates = pd.to_datetime(past_dates.reshape(-1),unit='s').to_numpy().reshape([-1,1])

            #Find the best matching f10.7 index along 2nd dimension
            match = np.argmin(np.abs(tmp_dates-self.f107[1].reshape([1,-1])),axis=-1)
            f107 = (self.f107[0][match]-self.scaler["omni"][0][-1])/(self.scaler["omni"][0][-1])
            past_omni = np.concatenate([po,dp.reshape(po.shape[0],1),f107.reshape(po.shape[0],1)],axis=-1)
            del po
            future_supermag = self.supermag_data[sg_ind[1],...][None,:]
            future_dates = np.array([self.dates[sg_ind[1]]])[None,:]
            sm_future = NamedAccess(future_supermag, self.supermag_features)
            _mlt = 90.0 - sm_future["MLT"] / 24.0 * 360.0
            _mcolat = 90.0 - sm_future["MAGLAT"]
            
            features_dict = {"past_omni": past_omni,
            "past_supermag": past_supermag,
            "future_supermag": future_supermag,
            "past_dates": past_dates,
            "future_dates": future_dates,
            "coords_radians": (np.deg2rad(_mlt), np.deg2rad(_mcolat))
            }
            self.sg_indices_list.append(features_dict)
            
    def compute_scaler(self, f107_data):
        self.scaler = {}
        print("learning scaler....")
        print("NOTE: Since the dataset is large, we take mean across only a limited set of samples due to memory constraint")
        N_SAMPLES = 10000
        np.random.seed(0)
        si = np.random.choice(len(self.sg_indices),size=int(N_SAMPLES/self.window_length),replace=False)
        sel_ind = self.sg_indices[si]
        new_inds = np.linspace(sel_ind[:,0],sel_ind[:,1],(sel_ind[:,1]-sel_ind[:,0])[0]).astype(int).T
        target = self.supermag_data[new_inds,...][...,self.target_idx]

        target_mean = np.nanmean(target, axis=(0,1,2))
        target_std = np.nanstd(target, axis=(0,1,2))
        self.scaler["supermag"] = [target_mean, target_std]

        dt=self.dates[new_inds]
        match =  np.argmin(np.abs(pd.to_datetime(dt.reshape(-1),unit='s').to_numpy().reshape(list(dt.shape))[...,None]\
                                  -f107_data["dates"].reshape([1,1,-1])),axis=-1)
        f107_tmp = f107_data["f107"][match]
        new_omni=np.concatenate([self.omni[i,...] for i in new_inds],axis=0)
        target = np.concatenate([new_omni,dipole_tilt(dt).reshape([-1,1]),f107_tmp.reshape([-1,1])],axis=-1)
        del new_omni
        omni_mean = np.nanmean(target, axis=0)
        omni_std = np.nanstd(target, axis=0)
        self.scaler["omni"] = [omni_mean, omni_std]
        # for i in self.sg_indices:
        print("During training time, all the supermag and original OMNI variables, including test set are normalized.")
        print("Hence, we don't need to normalize them again")
        print("So, during Wandb execution, the valures in val, weimer ds are all normalized. ")
        print("BUT THIS IS NOT THE CASE DURING STORM EXCECUTION IN SPACEML")
        self.supermag_data[...,self.target_idx] = (self.supermag_data[...,self.target_idx]-target_mean)/target_std

        self.omni = (self.omni-omni_mean[:-2])/omni_std[:-2]


    def process_data(self, i):
        sg_ind = self.sg_indices[i]
        po = self.omni[sg_ind[0]:sg_ind[0]+self.past_omni_length,...]
        past_supermag = self.supermag_data[sg_ind[0],...][None,:]
        past_dates = self.dates[sg_ind[0]:sg_ind[0]+self.past_omni_length]
        dp = (dipole_tilt(self.dates[sg_ind[0]:sg_ind[0]+self.past_omni_length])-self.scaler["omni"][0][-2])/(self.scaler["omni"][0][-2])
        tmp_dates = pd.to_datetime(past_dates.reshape(-1),unit='s').to_numpy().reshape([-1,1])

        #Find the best matching f10.7 index along 2nd dimension
        match = np.argmin(np.abs(tmp_dates-self.f107[1].reshape([1,-1])),axis=-1)
        f107 = (self.f107[0][match]-self.scaler["omni"][0][-1])/(self.scaler["omni"][0][-1])
        past_omni = np.concatenate([po,dp.reshape(po.shape[0],1),f107.reshape(po.shape[0],1)],axis=-1)
        del po
        future_supermag = self.supermag_data[sg_ind[1],...][None,:]
        future_dates = np.array([self.dates[sg_ind[1]]])[None,:]
        sm_future = NamedAccess(future_supermag, self.supermag_features)
        _mlt = 90.0 - sm_future["MLT"] / 24.0 * 360.0
        _mcolat = 90.0 - sm_future["MAGLAT"]

        features_dict = {"past_omni": past_omni,
        "past_supermag": past_supermag,
        "future_supermag": future_supermag,
        "past_dates": past_dates,
        "future_dates": future_dates,
        "coords_radians": (np.deg2rad(_mlt), np.deg2rad(_mcolat))
        }
        return features_dict
                                     
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="data_local/iaga/",year=yearlist))
omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=yearlist))
input_data = InputDataset(get_input_data(omni_path="data_local/omni/sw_data.h5",
                                             indices_path="data_local/supermag_indices/", 
                                             year=yearlist))

train_idx,test_idx,val_idx,wiemer_idx = generate_indices(base="data_local/iaga/",year=yearlist,
                                                        LENGTH=past_omni_length,LAG=lag,
                                                        omni_path="data_local/omni/sw_data.h5",
                                                        weimer_path="data_local/weimer/")
train_idx = np.asarray(train_idx)

train_ds_all_years = PreprocessData(supermag_data,input_data,train_idx,
                f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
                past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
                zero_supermag=False,scaler=None,training_batch=True,nmax=nmax, num_workers=num_workers)

scaler = train_ds_all_years.scaler
with open(os.path.join(output_folder, f'supermag_features.p'), 'wb') as f:
        pickle.dump(train_ds_all_years.supermag_features, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_folder, f'omni_features.p'), 'wb') as f:
    pickle.dump(train_ds_all_years.omni_features, f, pickle.HIGHEST_PROTOCOL)
    
with open(os.path.join(output_folder, f'scalers.p'), 'wb') as f:
    pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)
    
max_stations = get_iaga_max_stations(base="data_local/iaga/",yearlist=yearlist)
    
for year in yearlist:
    supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="data_local/iaga/",year=[year], max_stations=max_stations))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=[year]))
    input_data = InputDataset(get_input_data(omni_path="data_local/omni/sw_data.h5",
                                             indices_path="data_local/supermag_indices/", 
                                             year=[year]))

    train_idx,test_idx,val_idx,wiemer_idx = generate_indices(base="data_local/iaga/",year=[year],
                                                            LENGTH=past_omni_length,LAG=lag,
                                                            omni_path="data_local/omni/sw_data.h5",
                                                            weimer_path="data_local/weimer/")
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)
    wiemer_idx = np.asarray(wiemer_idx)
    
    train_ds = PreprocessData(supermag_data,input_data,train_idx,
                f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
                past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
                zero_supermag=False,scaler=scaler,training_batch=True,nmax=nmax, num_workers=num_workers)
    
    train_ds.compute_features()

    with open(os.path.join(output_folder, f'train_data_{year}.p'), 'wb') as f:
        pickle.dump(train_ds.sg_indices_list, f, pickle.HIGHEST_PROTOCOL)

    val_ds = PreprocessData(supermag_data,input_data,val_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=scaler,training_batch=False,nmax=nmax, num_workers=num_workers)
    
    val_ds.compute_features()

    with open(os.path.join(output_folder, f'val_data_{year}.p'), 'wb') as f:
        pickle.dump(val_ds.sg_indices_list, f, pickle.HIGHEST_PROTOCOL)


    wiemer_ds = PreprocessData(supermag_data,input_data,wiemer_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=scaler,training_batch=False,nmax=nmax, num_workers=num_workers)
    
    wiemer_ds.compute_features()

    with open(os.path.join(output_folder, f'wiemer_data_{year}.p'), 'wb') as f:
        pickle.dump(wiemer_ds.sg_indices_list, f, pickle.HIGHEST_PROTOCOL)
