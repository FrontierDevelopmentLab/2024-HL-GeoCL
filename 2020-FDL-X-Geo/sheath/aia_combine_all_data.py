import numpy as np
import zarr
import pandas as pd
import matplotlib.cm as cm
from astropy.time import Time
import dask.array as da
from glob import glob
import os
from tqdm import tqdm

"""
    We will need to map the OMNI dates to closest AIA dates. This allows us to form tuples of (AIA,swind). 
    Since there are many datapoints, we will use data at a lower cadence. 
"""

#====LOAD HMI DATA
# HMI time is in TAI, not UTC. So convert it first.
def stringchange(string):
    new_string = string.replace('_TAI', 'Z')
    new_string = new_string.replace('_','T')
    new_string = new_string.replace('.','-')
    new_string = new_string.replace('Z', '.00')
    return new_string
def convert_hmi_time_utc(time_hmi):
    t_obs_new = [stringchange(string) for string in time_hmi]
    t =  Time(t_obs_new, format='isot', scale='tai')
    t_obs_new=pd.to_datetime(t.utc.value,utc=True)
    return t_obs_new
#=============

years = np.arange(2010,2021).astype(int)
for year in tqdm(years):
    import pdb; pdb.set_trace()
    
    #Load SDOML data from v2_small
    AIAPATHS = sorted(glob(f"/mnt/disks/sdomlv2-full2/sdomlv2.zarr/{year}/*"))
    #Get passbands list
    PASSBANDS = [v.split('/')[-1] for v in AIAPATHS]
    #193 used for segmentation, so find its index.
    IND_193 = PASSBANDS.index("193A")
    # Get time stamps of all passbands, and find the nearest index to 193
    TIMES_AIA = [pd.to_datetime(zarr.open(v).attrs['T_OBS']) for v in AIAPATHS]
    
    #Get reference timestamp from the mask array
    times_aia_193 = np.load(f"/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/sheath_aia_data/AIA193_times_{year}.npy",allow_pickle=True)
    CLOSEST_AIA_INDICES = [np.argmin(np.abs(times_aia_193[None,...]-times[...,None]),axis=0) for times in TIMES_AIA]
    # This should be consistent with Generate_central_mask_CH.py
    NPIX = 17
    
    HMIPATHS = sorted(glob(f"/mnt/disks/sdomlv2-hmi2/sdomlv2_hmi.zarr/{year}/*"))
    #Get B components list
    BCOMP = [v.split('/')[-1] for v in HMIPATHS]
    # Get time stamps of all passbands, and find the nearest index to 193
    TIMES_HMI = [convert_hmi_time_utc(zarr.open(v).attrs['T_OBS']) for v in HMIPATHS]
    CLOSEST_HMI_INDICES = [np.argmin(np.abs(times_aia_193[None,...]-times[...,None]),axis=0) for times in TIMES_HMI]
    
    # Save these indices.
    np.savez("/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/logs/aia_hmi_inds_closest_to_193.npz", aia_inds = CLOSEST_AIA_INDICES,
             hmi_inds = CLOSEST_HMI_INDICES)
    
    # Multiply with mask and make a cube.
    
    CH_AR_mask = np.load(f"/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/sheath_aia_data/ch_mask_{year}.npy")
    ch_mask = CH_AR_mask[:,0,:]
    ar_mask = CH_AR_mask[:,1,:]
    
    feature_array = [ch_mask,ar_mask]
    #==== Subsample and get AIA dataset.
    for ind,path in enumerate(tqdm(AIAPATHS)):
        files = zarr.open(path)
        inds_aia = CLOSEST_AIA_INDICES[ind]
        files = np.asarray(files)[inds_aia,:,:]
        ch = files[:,:,256-NPIX:256+NPIX]*ch_mask
        ar = files[:,:,256-NPIX:256+NPIX]*ar_mask
        feature_array.append(ch)
        feature_array.append(ar)
        
    #==== Subsample and get HMI dataset. 
    for ind,path in enumerate(tqdm(HMIPATHS)):
        files = zarr.open(path)
        inds_hmi = CLOSEST_HMI_INDICES[ind]
        files = np.asarray(files)[inds_hmi,:,:]
        ch = files[:,:,256-NPIX:256+NPIX]*ch_mask
        ar = files[:,:,256-NPIX:256+NPIX]*ar_mask
        feature_array.append(ch)
        feature_array.append(ar)
    feature_array = np.asarray(feature_array)
    # Save this huge matrix!
    np.save(f"/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/sheath_aia_data/aia_subsamp_masked_{year}.npy",feature_array)
    

