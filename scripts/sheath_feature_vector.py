"""
This Python script extracts features from multi-wavelength observations of the Sun
collected by the Solar Dynamics Observatory (SDO). These features are then used
to generate a feature set for the SHEATH machine learning model, which
aims to predict the solar wind condition at L1 point.
"""

import csv
import os
import sys
from multiprocessing import Process, current_process

import pandas as pd
import tqdm as tq
import zarr

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))
from geocloak.preprocess.sdoprep import SDODataPreprocess  # noqa: E402

# Directory setup
# Later on it can be worked with the argparser

# Top level directories
rootdir = "/mnt/sdomlv2/sdomlv2a-static/"
sdodata = ["AIA.zarr", "HMI.zarr"]
out_dir = "/home/bjha/"
timeind_dir = "/home/bjha/data/geocloak/formatted_data/sdo/timestamps"


def sdoyear(year):
    """
    This helper function will generate the feature vector for the given year,
    and will be running on multiprocessor for parallelization. This function will
    store the extracted parameters in a csv file.

    Parameters
    ----------
    year : str
        The year for which the feature vector needs to be generated.
    """
    # AIA Data
    dataaia = zarr.open(os.path.join(rootdir, sdodata[0]))

    # HMI data
    datahmi = zarr.open(os.path.join(rootdir, sdodata[1]))

    yeardata_aia = dataaia[f"{year}"]
    yeardata_hmi = datahmi[f"{year}"]
    aiachannels = list(yeardata_aia.array_keys())
    hmifields = list(yeardata_hmi.array_keys())

    # Read Timeindex data
    timeindex = pd.read_csv(os.path.join(timeind_dir, f"time_index_{year}.csv"))
    timeindex.set_index("Times", inplace=True)
    timeindex.index = pd.to_datetime(timeindex.index)

    _outdir = os.path.join(out_dir, "sdoprepv2")
    if not os.path.exists(_outdir):
        os.makedirs(_outdir, exist_ok=True)

    filepath = os.path.join(_outdir, f"sdo_prep_{year}.csv")
    current_pro = current_process()._identity[0]
    with (
        open(filepath, "w") as csvfile,
        tq.tqdm(
            total=timeindex.shape[0],
            desc=f"{year}",
            position=current_pro,
        ) as pbar,
    ):
        for i, times in enumerate(timeindex.index):
            pbar.update()
            imagedict = {}
            for j, col in enumerate(timeindex.columns):
                if col in aiachannels:
                    imagedict[col] = yeardata_aia[col][timeindex.iloc[i, j], :, :]
                elif col in hmifields:
                    imagedict[col] = yeardata_hmi[col][timeindex.iloc[i, j], :, :]
            sdocls = SDODataPreprocess(imagedict, times, crop=False)
            feature_vector = sdocls.feature_vector()
            if i == 0:
                writer = csv.DictWriter(csvfile, fieldnames=feature_vector.keys())
                writer.writeheader()
            writer.writerow(feature_vector)


def main():
    years = [
        "2020",
        "2021",
        "2022",
        "2023",
    ]
    processes = []
    for year in years:
        p = Process(target=sdoyear, args=(year,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
