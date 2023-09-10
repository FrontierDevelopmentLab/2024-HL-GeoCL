import numpy as np
import zarr
import pandas as pd
import matplotlib.cm as cm
from astropy.time import Time
import dask.array as da
from glob import glob
import os
import multiprocessing as mp
from tqdm import tqdm

"""
    To pull the data, perform:
    gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_small.zarr ../sheath_data/sdoml_data/
    gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_hmi_small.zarr ../sheath_data/sdoml_data/
    
    And you will need opencv, zarr, dask, skimage for runnign this code.
"""



NPIX = 17
# Generate mask to get only the disc. I don't care about the limb
# Define solar disc
center = np.array([256,256])
radius = (695700.0/725.0)/(0.6*(4096/512.0)) #very approximate
print(f"Solar disc radius : {radius}")
xg = np.arange(0,512)
xgrid,ygrid = np.meshgrid(xg,xg)
distance = ((xgrid-center[0])**2+(ygrid-center[1])**2).astype(int)

#disc mask
mask = np.sign(distance-radius**2)
mask[mask>0] = np.nan
mask=np.abs(mask)

#===== Segementation code
#-- Algorithm here if interested: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020SW002478

from sklearn.mixture import GaussianMixture as GMM
import cv2
def GetMorphologicalStructure(og,mask,region=['AR'],n_comp=3):
    '''
        This function segments out the active regions, coronal holes and quiet sun from our images. It uses a Gaussian Mixture Model (GMM)
        to segement out the regions. GMM can be understood to be a generalization of Otsu thresholding.
        This function is a generalization of `GetActiveRegions`, where:
            1. Minimum of centroid mean corresponds to CHs.
            2. Maximum of centroid mean corresponds to ARs.
            3. Meidan of centroid mean corresponds to QS.
        Inputs:
            sample: img of shape [isize,isize], minvalue = 0 and maxvalue = 1
        
        This function is a part of suitpy package.
    '''
    #Initial smoothing
    # sample = cv2.bilateralFilter(og.astype(np.float32),9,75,75)
    sample = og*mask
    sample = sample[~np.isnan(sample)]
    
    #Define the mixture model, and take the component with highest mean value.
    gmodel = GMM(n_components=n_comp)
    gmodel.fit(np.reshape(sample,[-1,1]))
    th_gmm = gmodel.predict(np.reshape(og,[-1,1]))
    centroidfnlist = {'AR':np.max,"CH":np.min,"QS":np.median}
    assert all([True if x in ["AR","CH","QS"] else False for x in region]) 
    segments= {}
    mask
    for k in region:
        tmp = th_gmm == np.where(np.asarray(gmodel.means_)==centroidfnlist[k](gmodel.means_))[0]
        segments[k] = np.reshape(tmp,list(og.shape))
    return segments

def Get_CH_AR(og):
    ch = GetMorphologicalStructure(og,mask[:,256-NPIX:256+NPIX],region=['CH','AR'],n_comp=3)
    return np.asarray([ch["CH"],ch["AR"]])

# Load AIA 193 A data
sdomlv2_path = "/mnt/disks/sdomlv2-full2/sdomlv2.zarr/"
aia193_paths = sorted(glob(f"{sdomlv2_path}*/193A/"))

for path in tqdm(aia193_paths):
    print(path)
    year = path.split('/')[-3]
    sdomlsmall = zarr.open(path)
    print(f"Masking for: {path}, year: {year},size: {sdomlsmall.shape}")
    sdomlarr = list(np.log10(sdomlsmall[:,:,256-NPIX:256+NPIX]))
    pool = mp.Pool(processes=mp.cpu_count())
    ch_ar = np.asarray(pool.map(Get_CH_AR,sdomlarr))
    pool.close()
    mask[np.isnan(mask)] = 0.0
    ch_ar = ch_ar*(mask[:,256-NPIX:256+NPIX][None,...])
    
    SAVEPATH = "../sheath_aia_data/"
    if not os.path.isdir(SAVEPATH):
        os.makedirs(SAVEPATH)
    np.save(f"{SAVEPATH}ch_mask_{year}.npy",ch_ar)