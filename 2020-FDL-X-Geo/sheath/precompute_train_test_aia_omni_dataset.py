import pandas as pd
import numpy as np
from glob import glob



DATAPATH = "/home/jupyter/Vishal/clean_fdlx/2023-FDL-X-Geo/2020-FDL-X-Geo/sheath/"

# Load AIA data
list_aia = sorted(glob(f"{DATAPATH}sheath_aia_data/aia_subsamp_masked_summed_*.npy"))
aia_featureset =  np.concatenate([np.load(a) for a in list_aia],axis=-1).T

# Load AIA times
list_aia = sorted(glob(f"{DATAPATH}sheath_aia_data/AIA193_times_*.npy"))
aia_times = np.concatenate([np.load(a) for a in list_aia])
# Load OMNI data and dates
omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_complete.h5"
omni_data = pd.read_hdf(omni_path)
omni_data = omni_data[['Date', 'BX, nT (GSE, GSM)', 'BY, nT (GSM)', 'BZ, nT (GSM)', 'Speed, km/s', 
                       'Proton Density, n/cc', 'Proton Temperature, K']]
omni_inds_closest_to_aia = np.load(f"{DATAPATH}logs/closest_omni_aia_indices.npy")
omni_dates_closest_to_aia = np.load(f"{DATAPATH}logs/closest_omni_aia_dates.npy")

omni_data_closest_aia = omni_data.iloc[omni_inds_closest_to_aia]
#==== Now we have OMNI data corresponding to AIA data. We need to split this data into different datasets.

omni_data_closest_aia.to_hdf(f"{DATAPATH}logs/targets.h5",key="omni")
np.savez(f"{DATAPATH}logs/inputs.npz",data =aia_featureset, times = aia_times)