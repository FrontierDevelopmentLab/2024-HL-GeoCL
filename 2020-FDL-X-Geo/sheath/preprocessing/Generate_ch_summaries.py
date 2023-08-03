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
    To pull the data, perform:
    gsutil cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_small.zarr .
    gsutil cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_hmi_small.zarr .
    
    And you will need opencv, zarr, dask, skimage for runnign this code.
    
    CONSISTENCY of NPIX expected with Generate_central_mask_CH.py
"""

sdomlsmall = zarr.open("/home/jupyter/Vishal/sdoml/sdomlv2_small.zarr/2010/193A/")
times_193 = pd.to_datetime(sdomlsmall.attrs['T_OBS'])

# Load CH Mask
MASKPATH = "/home/jupyter/Vishal/sdoml_features/"
ch_mask = np.load(f"{MASKPATH}ch_mask.npy")

#Load SDOML data from v2_small
AIAPATHS = sorted(glob("/home/jupyter/Vishal/sdoml/sdomlv2_small.zarr/2010/*"))
#Get passbands list
PASSBANDS = [v.split('/')[-1] for v in AIAPATHS]
#193 used for segmentation, so find its index.
IND_193 = PASSBANDS.index("193A")
# Get time stamps of all passbands, and find the nearest index to 193
TIMES_AIA = [pd.to_datetime(zarr.open(v).attrs['T_OBS']) for v in AIAPATHS]
CLOSEST_AIA_INDICES = [np.argmin(np.abs(TIMES_AIA[IND_193][None,...]-times[...,None]),axis=0) for times in TIMES_AIA]
# This should be consistent with Generate_central_mask_CH.py
NPIX = 17
SORTED_INDICES_AIA_193 = np.argsort(TIMES_AIA[IND_193])


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

#Load HMI data from v2_small
HMIPATHS = sorted(glob("/home/jupyter/Vishal/sdoml/sdomlv2_hmi_small.zarr/2010/*"))
#Get B components list
BCOMP = [v.split('/')[-1] for v in HMIPATHS]
# Get time stamps of all passbands, and find the nearest index to 193
TIMES_HMI = [convert_hmi_time_utc(zarr.open(v).attrs['T_OBS']) for v in HMIPATHS]
CLOSEST_HMI_INDICES = [np.argmin(np.abs(TIMES_AIA[IND_193][None,...]-times[...,None]),axis=0) for times in TIMES_HMI]

SAVEPATH = "/home/jupyter/Vishal/sdoml_features/"
if not os.path.isdir(SAVEPATH):
    os.makedirs(SAVEPATH)
    
times_aia_save = np.asarray([v.to_numpy() for v in TIMES_AIA[IND_193]])[SORTED_INDICES_AIA_193]
np.save(f"{SAVEPATH}timestamps.npy",times_aia_save)

#==== Subsample and get AIA dataset. 
for ind,path in enumerate(tqdm(AIAPATHS)):
    files = zarr.open(path)
    inds_aia = CLOSEST_AIA_INDICES[ind]
    files = np.asarray(files)[inds_aia,:,:]
    files = files[:,:,256-NPIX:256+NPIX]*ch_mask
    files = files[SORTED_INDICES_AIA_193]
    np.save(f"{SAVEPATH}masked_{PASSBANDS[ind]}.npy",files)
    
#==== Subsample and get HMI dataset. 
for ind,path in enumerate(tqdm(HMIPATHS)):
    files = zarr.open(path)
    inds_hmi = CLOSEST_HMI_INDICES[ind]
    files = np.asarray(files)[inds_hmi,:,:]
    files = files[:,:,256-NPIX:256+NPIX]*ch_mask
    files = files[SORTED_INDICES_AIA_193]
    np.save(f"{SAVEPATH}masked_{BCOMP[ind]}.npy",files)