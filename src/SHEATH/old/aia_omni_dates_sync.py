"""
    This script syncs AIA and OMNI dates. We have the training and testing sets defined from OMNI data. Hence, we will map AIA dates to OMNI dates first.
    Then, we take only those OMNI time points which are closest to AIA time points, since the cadence is lower. And then from that set, we will need to split 
    the data into train-test-val sets.
"""

from glob import glob

import numpy as np
import pandas as pd

time_omni_backmapped = np.load("logs/sun_backtraced_dates.npy")

list_aia = sorted(glob("sheath_aia_data/AIA193_times_*.npy"))
time_aia = np.concatenate([np.load(a) for a in list_aia])

closest_omni_for_aia = np.asarray(
    [
        np.argmin(
            np.abs(pd.to_datetime(time_omni_backmapped) - pd.to_datetime(t)), axis=0
        )
        for t in time_aia
    ]
)
np.save("logs/closest_omni_aia_indices.npy", closest_omni_for_aia)
np.save("logs/closest_omni_aia_dates.npy", time_omni_backmapped[closest_omni_for_aia])
