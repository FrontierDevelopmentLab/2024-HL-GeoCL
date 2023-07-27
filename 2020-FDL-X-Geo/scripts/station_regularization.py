'''Script to compute scaling factors for SuperMAG stations.'''

import numpy as np
from pyproj import Geod
import matplotlib.pyplot as plt
from glob import glob
import re
import tqdm

base='../data_local/iaga/' # path to SuperMAG data
yearlist = list(np.arange(2010,2019).astype(int)) # years to compute scaling factors for

# get all SuperMAG data files
files = [g for y in yearlist for g in sorted(glob(f"{base}{y}/supermag_iaga_[!tiny]*.npz"),key=lambda f: int(re.sub("\D", "", f)),) ]

# get master SuperMAG station coords and store in dict
stn_coords = {}
with open('../supermag_stations.csv', 'r') as f:
    for line in f.readlines()[1:]:
        line = line.split(',')
        stn_coords[line[0]]=(float(line[1]), float(line[2])) # (IAGA,GEOLON,GEOLAT)

gg = Geod(ellps='WGS84') # distance calculator
Re = 6371.0 # Earth radius in km
nmax=100.0 # maximum degree of SpH used for relative scaling

# loop through all stations and get distance to all other stations
print("Computing scaling factors for SuperMAG stations...")
for i, f in enumerate(tqdm.tqdm(files)):
    dat = np.load(f, allow_pickle=True)
    dis_array = np.zeros((len(dat['stations']), len(dat['stations'])))
    for i in range(len(dat['stations'])):
        lon1,lat1 = stn_coords[dat['stations'][i]]
        for j in range(i,len(dat['stations'])):
            lon2,lat2 = stn_coords[dat['stations'][j]]
            if i==j: 
                dis_array[i,j]=np.nan
            else:
                dis_array[i,j]=gg.inv(lon1,lat1,lon2,lat2)[-1]/1000.0 #km
    dis_array = dis_array + dis_array.T # make symmetric
    # introduce linear scaling according to SpH order - defined through average distance of the 5 closest points
    dis=np.nanmean(np.sort(dis_array,axis=1)[:,:5],axis=1) # average distance of the 5 closest points
    lin_scaling=(np.pi*Re/dis)/nmax # linear scaling factor with SpH order relative to highest order nmax
    stn_scaling=np.nanmax(np.vstack([np.nanmin(np.vstack([lin_scaling,np.ones_like(dis)]),axis=0),np.ones_like(dis)*0.1]),axis=0) # generate floor and ceiling for scaling factor (relative to nmax)
    dat_len = dat['data'].shape[0]*np.ones_like(dis) # include length of data file to ensure reproducibility in terms of SuperMAG data features shape
    np.save(f.replace('.npz','_scaling.npy'),np.vstack([stn_scaling,dat_len])) # save scaling factor and length of dataset to file