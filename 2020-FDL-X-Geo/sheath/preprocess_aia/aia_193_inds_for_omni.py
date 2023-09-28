"""
    This script syncs AIA and OMNI dates. We have the training and testing sets defined from OMNI data. Hence, we will map AIA dates to OMNI dates first.
    We will select the AIA 193 data and which are closest to the backtraced OMNI data. This must be done for every year separately [for now!!]. And then from that set, we will need to split 
    the data into train-test-val sets.
"""
import pandas as pd
import numpy as np
from glob import glob
import zarr
from tqdm import tqdm

time_omni_backmapped = np.load("logs/sun_backtraced_dates.npy")
sdomlv2_path = "/mnt/disks/sdomlv2-full2/sdomlv2.zarr/"
times_pd = pd.to_datetime(time_omni_backmapped)
years = np.arange(2010,2021).astype(int)
for year in tqdm(years):
    inds = np.where(times_pd.year==year)
    sub_times = times_pd[inds]
    # Load AIA
    aia193_paths = f"{sdomlv2_path}{year}/193A/"
    sdomlsmall = zarr.open(aia193_paths)
    aia_times = pd.to_datetime(sdomlsmall.attrs['T_OBS'])
    aia_times = pd.to_datetime([t.replace(tzinfo=None) for t in aia_times])
    
    closest_omni_for_aia = np.argmin(np.abs(sub_times[...,None]-aia_times[None,...]),axis=1)
    np.save(f"logs/closest_omni_aia_indices_{year}.npy",closest_omni_for_aia)
    np.save(f"logs/closest_omni_aia_dates_{year}.npy",aia_times[closest_omni_for_aia])
                          
                          
                          