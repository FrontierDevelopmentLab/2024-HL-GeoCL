import os

import numpy as np
import pandas as pd

omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_1hour_full.h5"
omni_data = pd.read_hdf(omni_path)
dates = omni_data.Date.values

# years = np.arange(2010,2021).astype(int)
# datepath = "/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/logs/"
# dates = np.concatenate([np.load(f"{datepath}closest_omni_aia_dates_{year}.npy") for year in years])
print(dates[0], dates[-1])
N = len(dates)
print(f"Total wind measurements: {N}")
inds = np.arange(N).astype(int)

# First, some of the pre-defined dates will go into testing set. --
# NOTE: These timestamps are present in the data, check `Check_train_test_val_split.ipynb`

# 4 Aug 2011 storm
storm_2011_st = np.datetime64(pd.to_datetime("2011-08-04 00:00", unit="ns"))
storm_2011_end = np.datetime64(pd.to_datetime("2011-08-08 00:00", unit="ns"))
ind_2011_st = np.argmin(np.abs(dates - storm_2011_st))
ind_2011_en = np.argmin(np.abs(dates - storm_2011_end))
storm_2011 = inds[ind_2011_st:ind_2011_en]

# 16 March 2015 storm
storm_2015_st = np.datetime64(pd.to_datetime("2015-03-16 00:00", unit="ns"))
storm_2015_end = np.datetime64(pd.to_datetime("2015-03-20 00:00", unit="ns"))
ind_2015_st = np.argmin(np.abs(dates - storm_2015_st))
ind_2015_en = np.argmin(np.abs(dates - storm_2015_end))
storm_2015 = inds[ind_2015_st:ind_2015_en]

# 27 Sept 2017 storm
storm_2017_st = np.datetime64(pd.to_datetime("2017-09-25 00:00", unit="ns"))
storm_2017_end = np.datetime64(pd.to_datetime("2017-09-29 00:00", unit="ns"))
ind_2017_st = np.argmin(np.abs(dates - storm_2017_st))
ind_2017_en = np.argmin(np.abs(dates - storm_2017_end))
storm_2017 = inds[ind_2017_st:ind_2017_en]

# Now remaining indices will go in train-test-val.
N_test_done = len(storm_2015) + len(storm_2011) + len(storm_2017)
N_test_rem = int(N * 0.1) - N_test_done

N_val = int(0.2 * N)
N_train = int(0.7 * N)

taken = (
    np.arange(ind_2011_st, ind_2011_en).astype(int).tolist()
    + np.arange(ind_2015_st, ind_2015_en).astype(int).tolist()
    + np.arange(ind_2017_st, ind_2017_en).astype(int).tolist()
)
rem = [v for v in inds if v not in taken]

train_inds = inds[rem][:N_train]
val_inds = inds[rem][N_train : N_train + N_val]
test_inds = inds[rem][N_train + N_val : N_train + N_val + N_test_rem]
print(f"N train = {len(train_inds)}")
print(f"N val = {len(val_inds)}")
print(f"N test = {len(test_inds)}")
print(f"N 2011 = {len(storm_2011)}")
print(f"N 2015 = {len(storm_2015)}")
print(f"N 2017 = {len(storm_2017)}")
print(
    f"Ratios = {len(train_inds)/N:.2f}, {len(val_inds)/N:.2f}, {(len(test_inds)+len(storm_2015)+len(storm_2011)+len(storm_2017))/N:.2f}"
)

if not os.path.isdir("../logs/"):
    os.makedirs("../logs/")
np.savez(
    "../logs/sheath_train_test_val_split.npz",
    storm_2015=storm_2015,
    storm_2017=storm_2017,
    storm_2011=storm_2011,
    train=train_inds,
    val=val_inds,
    test=test_inds,
)
