import copy
import multiprocessing as mp
import os
import sys
from glob import glob

import dask.array as da
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import zarr
from astropy.time import Time
from tqdm import tqdm

sys.path.append("../")
from utils.preprocessing import GetMorphologicalStructure, mask_from_aia_193


# ==== Define a wrapper on top.
def Get_CH_AR(og):
    ch = GetMorphologicalStructure(og, mask, region=["CH", "AR"], n_comp=3)
    return np.asarray([ch["CH"], ch["AR"]])


"""
    To pull the data, perform:
    gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_small.zarr ../sheath_data/sdoml_data/
    gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_hmi_small.zarr ../sheath_data/sdoml_data/
    
    And you will need opencv, zarr, dask, skimage for runnign this code.
"""

NPIX = 17
# Generate mask to get only the disc. I don't care about the limb
# Define solar disc
center = np.array([256, 256])
radius = (695700.0 / 725.0) / (0.6 * (4096 / 512.0))  # very approximate
print(f"Solar disc radius : {radius}")
xg = np.arange(0, 512)
xgrid, ygrid = np.meshgrid(xg, xg)
distance = ((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2).astype(int)

# disc mask
mask = np.sign(distance - radius**2)
mask[mask > 0] = np.nan
mask = np.abs(mask)
mask = mask[:, 256 - NPIX : 256 + NPIX]

# Load AIA 193 A data
sdomlv2_path = "/mnt/disks/sdomlv2-full2/sdomlv2.zarr/"
aia193_paths = sorted(glob(f"{sdomlv2_path}*/193A/"))
SUB = 10  # 1 sample every hour.

mask_new = copy.deepcopy(mask)
mask_new[np.isnan(mask)] = 0.0

for path in tqdm(aia193_paths):
    year = path.split("/")[-3]
    sdomlsmall = zarr.open(path)
    """
        We do not need the full time cadence of SDOML. Our model can work with 1 hour cadence 
        for this work. While inference we can perform on any minute data, since we do not have 
        temporal dependence.
    """
    # Get time stamps
    timestamps = pd.to_datetime(sdomlsmall.attrs["T_OBS"])
    # These timestamps are NOT sorted. We wil need to sort them to be in order
    timestamps = pd.to_datetime([t.replace(tzinfo=None) for t in timestamps])
    indices = np.load(f"../logs/closest_omni_aia_indices_{year}.npy")
    # We need these new times to map other AIA and HMI data here.

    # Subsample in space
    sdomlsmall = sdomlsmall[:, :, 256 - NPIX : 256 + NPIX]
    # Sort in time
    sdomlsmall = sdomlsmall[indices]
    # db = timestamps[ind_sorted_times]
    # Calculate delta T and assert sorting is done, as a sanity check.
    # db = db[1:]-db[:-1]
    # assert len(np.where(db.total_seconds()<0)[0])==0

    # Subsample in time
    # sdomlsmall = sdomlsmall[::SUB]
    # new_times = new_times[::SUB]

    print(f"Masking for: {path}, year: {year},size: {sdomlsmall.shape}")
    timestamps = timestamps[indices]

    new_times, ind_sorted_times = timestamps.sort_values(return_indexer=True)
    sdomlarr = list(mask_from_aia_193(sdomlsmall))
    pool = mp.Pool(processes=mp.cpu_count())
    ch_ar = np.asarray(pool.map(Get_CH_AR, sdomlarr))
    pool.close()

    ch_ar = ch_ar * (mask_new[None, ...])

    SAVEPATH = "../sheath_aia_data/"
    if not os.path.isdir(SAVEPATH):
        os.makedirs(SAVEPATH)
    np.save(f"{SAVEPATH}ch_mask_{year}.npy", ch_ar)
    np.save(f"{SAVEPATH}AIA193_times_{year}.npy", timestamps)
