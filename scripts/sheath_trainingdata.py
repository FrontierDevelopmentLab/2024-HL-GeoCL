#!/usr/bin/env python
"""
This is the python scrpit which maps the OMNI/ACE/DSCOVR time to SDO time.
Using the method (i) Ballistic back propagation, (ii) HUX and use the backtracked
information to create the training data set for SHEATH.
"""

import pandas as pd
import numpy as np
import astropy.constants as const
import astropy.units as u
import os
from astropy.time import Time
import tqdm as tq
import csv
from sunpy.coordinates.sun import (
    L0,
    carrington_rotation_number,
    carrington_rotation_time,
)


def ballistic(times, velocity):
    """
    Thi function will map the give time and velocity to SDO time
    using the ballistic back propogation approach.

    Parameters
    ----------
    times : datetime | str
        The time for which the SDO time needs to be calculated.
    velocity : float
        The velocity of the solar wind at the given time.

    Returns
    -------
    sdotime : datetime
        The SDO time corresponding to the given time and velocity.
    """
    try:
        times = pd.to_datetime(times)
    except TypeError:
        raise TypeError("Unable to Conver time to datetime.")

    dt = (const.au.to("km").value / velocity) / (3600.0 * 24)
    newtime = (times - pd.to_timedelta(dt, unit="day")).round("min")
    return newtime


# Top level variables for the run
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAXIMUM_TIME_DIFF = 1  # days

# Location for SDO preprocessed data
sdo_dir = "/home/bjha/data/geocloak/formatted_data/sdo/sdoprep/"

# Location for OMNI data
omni_dir = "/home/bjha/data/geocloak/formatted_data/OMNI"

# Output Directory
out_dir = "/home/bjha/data/geocloak/formatted_data/sheath_train"

# Backtracking method
backtrack_method = ballistic

# List of years need to be run (make sure that data are available)
years = [2011, 2012, 2013]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("%" * 50)
print(f'Generating SHEATH data usning "{backtrack_method.__name__}" for back-tracking.')

for year in tq.tqdm(years, position=0, leave=True, desc=f"Total"):

    # Outfile path/name
    filepath = os.path.join(out_dir, f"sheath_training_{year}.csv")

    # Read data
    omnidf = pd.read_hdf(os.path.join(omni_dir, f"omniweb_formatted_{year}.h5"))
    omnidf.index = pd.to_datetime(omnidf.index)
    sdodf = pd.read_csv(os.path.join(sdo_dir, f"sdo_prep_{year}.csv"), index_col=0)
    sdodf.index = pd.to_datetime(sdodf.index)

    # Create header for CSV writer
    fieldnames = ["Time"] + list(sdodf.columns) + list(omnidf.columns)

    # Strat back tracking using specified method and writing data
    with open(filepath, "w") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        # Main loop
        for _time in tq.tqdm(omnidf.index, position=1, desc=f"{year}", leave=False):
            if not np.array(omnidf.loc[_time]).all():
                continue
            _sdotime = backtrack_method(_time, omnidf.loc[_time].Speed)
            ind = np.nanargmin(sdodf.index - _sdotime)
            dt = sdodf.index[ind] - _sdotime
            if dt.days > MAXIMUM_TIME_DIFF:
                continue

            # Write data
            outdict = {"Time": _time.isoformat()}
            outdict.update(sdodf.iloc[ind].to_dict())
            outdict.update(omnidf.loc[_time].to_dict())
            csvwriter.writerow(outdict)
print("Completed")
print(f"All data are saved in {out_dir}.")
print("%" * 50)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
