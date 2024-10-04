#!/usr/bin/env python
"""
This is the python scrpit which maps the OMNI/ACE/DSCOVR time to SDO time.
Using the method (i) Ballistic back propagation, (ii) HUX and use the backtracked
information to create the training data set for SHEATH.
"""

import csv
import os
import sys
from multiprocessing import Process, current_process

import numpy as np
import pandas as pd
import tqdm.auto as tq

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))
from geocloak.preprocess.backtrack import ballistic  # noqa: E402

# Top level variables for the run
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAXIMUM_TIME_DIFF = 1  # days

# Location for SDO preprocessed data
sdo_dir = "/home/bjha/sdoprepv2/"

# Location for OMNI data
omni_dir = "/home/bjha/data/geocloak/formatted_data/OMNI/omniweb_1m"

# Output Directory
out_dir = "/home/bjha/data/geocloak/formatted_data/sheath_train"

# Backtracking method
backtrack_method = ballistic

# List of years need to be run (make sure that data are available)
years = [2011, 2012, 2013]

# One Carrington Rotation
dindex = int(27.3 * 24 * 60)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cadence = "1m"


# for year in tq.tqdm(years, position=0, leave=True, desc=f"Total"):
def sheath_training(year):

    current_pro = current_process()._identity[0]

    # Outfile path/name
    filepath = os.path.join(
        out_dir, f"sheath_training_{cadence}_{backtrack_method.__name__}_{year}.csv"
    )

    # Read data
    omnidf = pd.read_hdf(
        os.path.join(omni_dir, f"omniweb_formatted_{cadence}_{year}.h5")
    )
    omnidf.index = pd.to_datetime(omnidf.index)
    omnidf.dropna(inplace=True)

    sdodf2 = pd.read_csv(os.path.join(sdo_dir, f"sdo_prep_{year}.csv"), index_col=0)
    if os.path.exists(os.path.join(sdo_dir, f"sdo_prep_{year-1}.csv")):
        sdodf1 = pd.read_csv(
            os.path.join(sdo_dir, f"sdo_prep_{year-1}.csv"), index_col=0
        )
        sdodf = pd.concat([sdodf1, sdodf2])
    else:
        sdodf = sdodf2
    sdodf.index = pd.to_datetime(sdodf.index)

    # Create header for CSV writer
    fieldnames = ["Time", "TimeSDO"] + list(sdodf.columns) + list(omnidf.columns)

    # Strat back tracking using specified method and writing data
    with open(filepath, "w") as csvfile, tq.tqdm(
        total=omnidf.shape[0], position=current_pro, desc=f"{year}"
    ) as pbar:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for ii, _time in enumerate(omnidf.index):
            pbar.update()
            if not np.array(omnidf.loc[_time]).all():
                continue
            if backtrack_method.__name__ == "ballistic":
                _sdotime = backtrack_method(_time, omnidf.loc[_time].Speed)
            elif backtrack_method.__name__ == "HUX":
                _sdotime = backtrack_method(
                    omnidf.index[ii - dindex : ii + 1],
                    omnidf.Speed.values[ii - dindex : ii + 1],
                )

            ind = np.nanargmin(np.abs(sdodf.index - _sdotime).total_seconds())
            dt = sdodf.index[ind] - _sdotime
            if dt.total_seconds() / 86400.0 > MAXIMUM_TIME_DIFF:
                continue

            # Write data
            outdict = {
                "Time": _time.isoformat(),
                "TimeSDO": sdodf.index[ind].isoformat(),
            }
            outdict.update(sdodf.iloc[ind].to_dict())
            outdict.update(omnidf.loc[_time].to_dict())
            csvwriter.writerow(outdict)

    dfraw = pd.read_csv(filepath)
    dfraw.Time = pd.to_datetime(dfraw.Time)
    dmean = dfraw.groupby("TimeSDO").mean()
    dmean = dfraw.groupby("TimeSDO").mean()
    dstd = (
        dfraw[["TimeSDO", "Speed", "Density", "Temperature", "Bt", "Bx", "By", "Bz"]]
        .groupby("TimeSDO")
        .std(ddof=0)
    )
    dstd.rename(lambda x: x + "sd", axis=1, inplace=True)

    filepath = os.path.join(
        out_dir,
        f"sheath_training_mstd_{cadence}_{backtrack_method.__name__}_{year}.csv",
    )
    df = dmean.join(dstd, how="inner")
    df.to_csv(filepath)


def main():
    years = range(2015, 2020)
    processes = []
    for year in years:
        p = Process(target=sheath_training, args=(year,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    print("%" * 50)
    print(
        f'Generating SHEATH data usning "{backtrack_method.__name__}" for back-tracking.'
    )
    main()
    print("Completed")
    print(f"All data are saved in {out_dir}.")
    print("%" * 50)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
