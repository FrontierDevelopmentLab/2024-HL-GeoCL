import numpy as np
from glob import glob 
import pandas as pd  
import datetime
import sys
import os

omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_complete.h5"
omni_data = pd.read_hdf(omni_path)
dates = omni_data.Date.values
N = omni_data.shape[0]
print(f"Total wind measurements: {N}")
inds = np.arange(N).astype(int)

##--First, some of the pre-defined dates will go into testing set. --
# NOTE: These timestamps are present in the data, check `Check_train_test_val_split.ipynb`

#Brooks+ 2014
storm_2011_st = np.datetime64(pd.to_datetime("2011-08-04 00:00",unit='ns'))
storm_2011_end = np.datetime64(pd.to_datetime("2011-08-08 00:00",unit='ns'))
ind_2011_st = np.argmin(np.abs(dates-storm_2011_st))
ind_2011_en = np.argmin(np.abs(dates-storm_2011_end))
storm_2011 = inds[ind_2011_st:ind_2011_en]

#PSP 2020
storm_2015_st = np.datetime64(pd.to_datetime("2015-03-16 00:00",unit='ns'))
storm_2015_end = np.datetime64(pd.to_datetime("2015-03-20 00:00",unit='ns'))
ind_2015_st = np.argmin(np.abs(dates-storm_2015_st))
ind_2015_en = np.argmin(np.abs(dates-storm_2015_end))
storm_2015 = inds[ind_2015_st:ind_2015_en]

#Now remaining indices will go in train-test-val.
N_test_done = len(storm_2015)+len(storm_2011)
N_test_rem = int(N*0.1)-N_test_done

N_val = int(0.2*N)
N_train = int(0.7*N)

taken = np.arange(ind_2011_st,ind_2011_en).astype(int).tolist()+np.arange(ind_2015_st,ind_2015_en).astype(int).tolist()
rem = [v for v in inds if v not in taken]

train_inds = inds[rem][:N_train]
val_inds = inds[rem][N_train:N_train+N_val]
test_inds = inds[rem][N_train+N_val:N_train+N_val+N_test_rem]
print(f"N train = {len(train_inds)}")
print(f"N val = {len(val_inds)}")
print(f"N test = {len(test_inds)}")
print(f"N 2011 = {len(storm_2011)}")
print(f"N 2015 = {len(storm_2015)}")
print(f"Ratios = {len(train_inds)/N:.2f}, {len(val_inds)/N:.2f}, {(len(test_inds)+len(storm_2015)+len(storm_2011))/N:.2f}")

if not os.path.isdir("../logs/"):
    os.makedirs("../logs/")
np.savez("../logs/sheath_train_test_val_split.npz",storm_2015=storm_2015,
            storm_2011=storm_2011,train=train_inds,val=val_inds,test=test_inds)