#!/usr/bin/env python

import zarr
import pandas as pd
import numpy as np
from astropy.time import Time
import tqdm as tq
import os

"""
This is the python script which will generate the time index file
used to generate the feature set for SHEATH by combining the multi
wavelength observation from the SDO.
"""

# Directory setup
# Later on it can be worked with the argparser

# Top level directories
rootdir = "/mnt/sdoml/"
sdodata = ["AIA.zarr", "HMI.zarr"]
out_dir = "/home/bjha/data/geocloak/formatted_data/sdo"

# List of years
years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]

# AIA Data
dataaia = zarr.open("/mnt/sdoml/" + sdodata[0])

# HMI data
datahmi = zarr.open("/mnt/sdoml/" + sdodata[1])

for year in years:

    # HMI as reference data
    print(f"Generating Reference timestamps from HMI for {year}.")
    print("%" * 60)
    hmi720 = datahmi[f"{year}/Bz"]
    hmi720times = pd.to_datetime(
        hmi720.attrs["T_OBS"], format="%Y.%m.%d_%H:%M:%S.%f_TAI"
    )
    hmi720times = Time(np.sort(hmi720times))

    allind = {"Times": hmi720times.fits}
    alltimes = {"Times": hmi720times.fits}

    # AIA data
    channels = dataaia[f"{year}"]
    for ch in channels.array_keys():
        _times = Time(np.sort(channels[ch].attrs["T_OBS"]))
        ind = []
        for result in tq.tqdm(hmi720times, desc=f"{year:4}/{ch:6}"):
            _ind = np.argmin(np.abs(_times.jd - result.jd))
            ind.append(_ind)
        allind[ch] = np.array(ind)
        alltimes[ch] = _times[ind].fits

    # HMI data
    channels = datahmi[f"{year}"]
    for ch in channels.array_keys():
        formated_time = pd.to_datetime(
            channels[ch].attrs["T_OBS"], format="%Y.%m.%d_%H:%M:%S.%f_TAI"
        )
        _times = Time(np.sort(formated_time))
        ind = []
        for result in tq.tqdm(hmi720times, desc=f"{year:4}/{ch:6}"):
            _ind = np.argmin(np.abs(_times.jd - result.jd))
            ind.append(_ind)
        allind[ch] = np.array(ind)
        alltimes[ch] = _times[ind].fits
    _outdir = os.path.join(out_dir, "timestamps")

    # Save data as csv file
    if not os.path.exists(_outdir):
        os.makedirs(_outdir, exist_ok=True)
    indexfilename = f"time_index_{year}.csv"
    timestampfilename = f"timestamp_index_{year}.csv"
    print(f"Saving {indexfilename} in {_outdir}.")
    pd.DataFrame(allind).to_csv(os.path.join(_outdir, indexfilename), index=False)

    print(f"Saving {timestampfilename} in {_outdir}.")
    pd.DataFrame(alltimes).to_csv(os.path.join(_outdir, timestampfilename), index=False)

    print("Completed")
    print("%" * 60)
