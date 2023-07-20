import re
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm
from scipy.special import sph_harm
from torch.utils import data
import h5py
from glob import glob
from astropy.time import Time
import sys

sys.path.append('../')
from dataloader import (OMNIDataset, ShpericalHarmonicsDataset,
                        SuperMAGIAGADataset)


def persist_to_file(file_name):
    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(param):
            if param not in cache:
                cache[param] = original_func(param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[param]

        return new_func

    return decorator


def get_omni_data(path=None, year="2016"):
    import pandas as pd
    if isinstance(year,str):
        return pd.read_hdf(path, key=year)
    elif isinstance(year,list):
        return pd.concat([pd.read_hdf(path, key=str(y)) for y in year])
    else:
        raise TypeError("year must be either a list of years, or a single year.")


def get_iaga_max_stations(base="data_local/iaga/", yearlist=[2013], tiny=False):
    if tiny:
        files = [g for y in yearlist for g in sorted(glob(f"{base}{y}/supermag_iaga_tiny*.npz"),key=lambda f: int(re.sub("\D", "", f)),) ]
    else:
        files = [g for y in yearlist for g in sorted(glob(f"{base}{y}/supermag_iaga_[!tiny]*.npz"),key=lambda f: int(re.sub("\D", "", f)),) ]
    assert len(files) > 0
    stations = []

    print("Loading SuperMAG IAGA data to determine maximum number of stations...")
    for i, f in enumerate(tqdm.tqdm(files)):
        x = np.load(f, allow_pickle=True)
        stations.append(x["stations"])

    max_stations = max([len(s) for s in stations])
    return max_stations

def get_iaga_data_as_list(base,year,tiny=False,load_data=True,stn_reg=False):
    if isinstance(year,str):
        dates,data,features = get_iaga_data(f"{base}{year}/",tiny=tiny,load_data=load_data)
        if stn_reg:
            reg=get_iaga_reg(f"{base}{year}/")
            return dates,data,features,reg
        else:
            return dates,data,features,np.ones_like(data)

    elif isinstance(year,list):
        dates = []
        data = []
        features = []
        reg = []
        max_stations = get_iaga_max_stations(base=base, yearlist=year)
        for y in year:
            dt,dat,feat = get_iaga_data(f"{base}{y}/",tiny=tiny,max_stations=max_stations,load_data=load_data)
            dates.append(dt)
            features.append(feat)
            if load_data:
                data.append(dat)
                if stn_reg: 
                    reg.append(get_iaga_reg(f"{base}{y}/",max_stations=max_stations))
                else:
                    reg.append(np.ones_like(dat))
        dates = np.concatenate(dates,axis=0)

        if load_data:
            data = np.concatenate(data,axis=0)
            reg = np.concatenate(reg,axis=0)
        
        return dates,data,features,reg
    
    else:
        raise TypeError("year must be either a list of years, or a single year.")


def get_iaga_data(path, tiny=False, load_data=True,max_stations=None):

    import tqdm

    if tiny:
        files = sorted(
            [f for f in glob(path + "supermag_iaga_tiny*.npz")],
            key=lambda f: int(re.sub("\D", "", f)),
        )
    else:
        files = sorted(
            [f for f in glob(path + "supermag_iaga_[!tiny]*.npz")],
            key=lambda f: int(re.sub("\D", "", f)),
        )
    assert len(files) > 0

    data = []
    dates = []
    stations = []
    # idx = []

    print(f"Loading SuperMAG IAGA data: {path}")
    for i, f in enumerate(tqdm.tqdm(files)):
        x = np.load(f, allow_pickle=True)
        if load_data:
            data.append(x["data"])
        dates.append(x["dates"])
        # print(np.datetime64(datetime.utcfromtimestamp(dates[-1][0])))
        #idx.extend(data[-1].shape[0] * [i])
        features = x["features"]
        stations.append(x["stations"])

    if max_stations is None:
        max_stations = max([len(s) for s in stations])
    else:
        max_stations = max_stations
    if load_data:
        for i, d in enumerate(data):
            data[i] = np.concatenate(
                [d, np.zeros([d.shape[0], max_stations - d.shape[1], d.shape[2]]) * np.nan],
                axis=1,
            )
        data = np.concatenate(data,axis=0)
    dates = np.concatenate(dates)

    return dates, data, features

def get_iaga_reg(base,max_stations=None):

    import tqdm

    files = sorted(
        [f for f in glob(base + "supermag_iaga_[!tiny]*_scaling.npy")],
        key=lambda f: int(re.sub("\D", "", f)),
    )
    assert len(files) > 0

    sca_dat = []
    stations_len =[]
    print(f"Loading SuperMAG scaling data: {base}")
    for i, f in enumerate(tqdm.tqdm(files)):
        x = np.load(f, allow_pickle=True)
        sca_all = np.expand_dims(np.tile(x[0,:],(int(x[1,0]),1)),axis=2)
        sca_dat.append(sca_all)
        stations_len.append(len(x[0,:]))

    if max_stations is None:
        max_stations = np.max(np.array(stations_len))
    else:
        max_stations = max_stations
        
    for i, d in enumerate(sca_dat):
        sca_dat[i] = np.concatenate(
            [d, np.zeros([d.shape[0], max_stations - d.shape[1], d.shape[2]]) * np.nan],
            axis=1,
        )
    sca_dat = np.concatenate(sca_dat,axis=0)

    return sca_dat

def get_weimer_data_indices(targets, lag, past_omni_length, future_length,sg_data,weimer_years):
    """
        Function to return the indices corresponding to storm times.
        We have 3 storms for Weimer data: 2011, 2015 and 2017. These 3 will be in weimer/ folder.
        This code takes in the already loaded supermag measurements, and finds the correct times of
        weimer predictions. These form our test set.

        Code needs either a single year as a string, or a list of years for generating the indices.

        USE THIS CODE TO REMOVE THE INDICES CORRESPONDING TO WEIMER DATA (IF ANY) FROM THE TRAINING
        SET.
    """
    if isinstance(weimer_years,str):
        weimer_years = [weimer_years]

    if isinstance(weimer_years,list):
        storm_inds = []
        for year in weimer_years:
            weimer = {}
            fpath = glob(f"data_local/weimer/TimeStepGeomagnetic_{year}*.h5")[0]
            with h5py.File(fpath, "r") as f:
                for k in f.keys():
                    weimer[k] = f.get(k)[:]
            weimer_times_unix = Time(weimer["JDTIMES"], format="jd").to_value("unix")
            wstart = np.argmin(np.abs(weimer_times_unix[0] - sg_data.dates)) - past_omni_length -lag - future_length +2
            wend = (np.argmin(np.abs(weimer_times_unix[-1] - sg_data.dates)) + 1)
            weimerinds = np.arange(wstart, wend).astype(int)
            if len(weimerinds)>1:
                storm_inds.append(weimerinds)
        return np.concatenate(storm_inds,axis=0)
    else:
        raise TypeError("Weimer year must be either a list of years, or a single year.")

def get_wiemer_data(omni_data, supermag_data, targets, scaler, lag, past_omni_length, future_length,wyear):
    """
        Function to load the OMNI and SuperMAG measurements corresponding to Weimer storm time.
        NOTE:::::!!This function actually load the data, and not just returns the indices.
    """

    if isinstance(wyear,str):
        wyear = [wyear]
    if isinstance(wyear,list):
        datasets = {}
        for year in wyear:
            weimer = {}
            fpath = glob(f"data_local/weimer/TimeStepGeomagnetic_{year}*.h5")[0]
            with h5py.File(fpath, "r") as f:
                for k in f.keys():
                    weimer[k] = f.get(k)[:]

            weimer_times_unix = Time(weimer["JDTIMES"], format="jd").to_value("unix")
            wstart = np.argmin(np.abs(weimer_times_unix[0] - supermag_data.dates)) - past_omni_length -lag - future_length +2
            wend = (np.argmin(np.abs(weimer_times_unix[-1] - supermag_data.dates)) + 1)
            weimerinds = np.arange(wstart, wend).astype(int)
            datasets[year] = (ShpericalHarmonicsDataset(
                                                        supermag_data,
                                                        omni_data,
                                                        weimerinds,
                                                        scaler=scaler,
                                                        targets=targets,
                                                        past_omni_length=past_omni_length,
                                                        future_length=future_length,
                                                        f107_dataset="data_local/f107.npz"
                                                    ))
        return datasets
    else:
        raise TypeError("Weimer year must be either a list of years, or a single year.")


def load_cached_data(filename, idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length):
    if os.path.exists(filename):
        data = pickle.load(open(filename, "rb"))
        return data, data.scaler
    else:
        data = ShpericalHarmonicsDataset(
            supermag_data,
            omni_data,
            idx,
            scaler=scaler,
            targets=targets,
            past_omni_length=past_omni_length,
            future_length=future_length,
            f107_dataset="data_local/f107.npz",
        )
        directorypath = '/'.join(filename.split('/')[:-1])
        if not os.path.isdir(directorypath):
            os.makedirs(directorypath)
        pickle.dump(data, open(filename, "wb"))
        return data, data.scaler