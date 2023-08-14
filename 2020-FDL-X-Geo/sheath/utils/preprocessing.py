import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import cv2
import sys
from utils.torch_utils import _float
import xgboost as xgb
import matplotlib.pyplot as plt


def get_mask(n_size = 512.0):
    # Generate mask to get only the disc. I don't care about the limb
    # Define solar disc
    center = np.array([256,256])
    radius = (695700.0/725.0)/(0.6*(4096/n_size)) #very approximate
    xg = np.arange(0,n_size)
    xgrid,ygrid = np.meshgrid(xg,xg)
    distance = ((xgrid-center[0])**2+(ygrid-center[1])**2).astype(int)

    #disc mask
    mask = np.sign(distance-radius**2)
    mask[mask>0] = np.nan
    mask=np.abs(mask)
    return mask

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
    for k in region:
        tmp = th_gmm == np.where(np.asarray(gmodel.means_)==centroidfnlist[k](gmodel.means_))[0]
        segments[k] = np.reshape(tmp,list(og.shape))
    return segments

def subsample_and_segment(og, mask, region = ['CH'], ncomp = 3):
    segmentation = GetMorphologicalStructure(og,mask,region=region,n_comp=ncomp)
    return segmentation

class Preprocessor_CH:
    def __init__(self, npix = 17, n_size = 512, ch_mask = True, scaler_aia = None):
        self.ch_mask = ch_mask
        self.scaler_aia = scaler_aia
        self.npix = npix
        self.n_size = n_size
        
    def preprocess(self, aia_data, hmi_data, aia_wavelengths, hmi_components):
        ind_193 = aia_wavelengths.index('193A')
        n_size = self.n_size
        npix = self.npix
        mask = get_mask(n_size = n_size)[:,int(n_size//2)-npix:int(n_size//2)+npix]
        if self.ch_mask:
            segmentation = subsample_and_segment(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix], mask,
                                                 region = ['CH'], ncomp = 3)
            segmentation = segmentation["CH"]
        else:
            segmentation = np.ones_like(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix])
    
        aia_data = np.asarray([aia_data[channel][:,int(n_size//2)-npix:int(n_size//2)+npix] for channel in aia_data.keys()])*(segmentation[None,...])
        hmi_data = np.asarray([hmi_data[comp][:,int(n_size//2)-npix:int(n_size//2)+npix] for comp in hmi_data.keys()])*(segmentation[None,...])
        # The input data should be of the form [512,34,12]
        in_data = np.concatenate([aia_data,hmi_data]).transpose([1,2,0])[None,...]
        if self.scaler_aia is not None:
            in_data = self.scaler_aia.transform(in_data)
        # Should return a torch tensor of form [None,512,34,channel]
        return _float(in_data)
    
    
class Preprocessor_CH_xgb:
    def __init__(self, npix = 17, n_size = 512, ch_mask = True, scaler_aia = None):
        self.ch_mask = ch_mask
        self.scaler_aia = scaler_aia
        self.npix = npix
        self.n_size = n_size
        
    def preprocess(self, aia_data, hmi_data, aia_wavelengths, hmi_components):
        ind_193 = aia_wavelengths.index('193A')
        n_size = self.n_size
        npix = self.npix
        mask = get_mask(n_size = n_size)[:,int(n_size//2)-npix:int(n_size//2)+npix]
        if self.ch_mask:
            ch_segmentation = subsample_and_segment(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix], mask,
                                                 region = ['CH'], ncomp = 3)
            ch_segmentation = ch_segmentation["CH"]
            ar_segmentation = subsample_and_segment(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix], mask,
                                                 region = ['AR'], ncomp = 3)
            ar_segmentation = ar_segmentation["AR"]
        else:
            ch_segmentation = np.ones_like(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix])
            ar_segmentation = np.ones_like(aia_data["193A"][:,int(n_size//2)-npix:int(n_size//2)+npix])
        # plt.imshow(aia_data["193A"])
        # plt.savefig("sheath_data/193A_movie/"
        aia_ch_data = np.asarray([aia_data[channel][:,int(n_size//2)-npix:int(n_size//2)+npix] for channel in aia_data.keys()])\
            *(ch_segmentation[None,...])
        aia_ar_data = np.asarray([aia_data[channel][:,int(n_size//2)-npix:int(n_size//2)+npix] for channel in aia_data.keys()])\
        *(ar_segmentation[None,...])
        hmi_ch_data = np.asarray([hmi_data[comp][:,int(n_size//2)-npix:int(n_size//2)+npix] for comp in hmi_data.keys()])*(ch_segmentation[None,...])
        hmi_ar_data = np.asarray([hmi_data[comp][:,int(n_size//2)-npix:int(n_size//2)+npix] for comp in hmi_data.keys()])*(ar_segmentation[None,...])
        ch_dataset = np.concatenate([aia_ch_data,hmi_ch_data]).transpose([1,2,0])[None,...]
        ar_dataset = np.concatenate([aia_ar_data,hmi_ar_data]).transpose([1,2,0])[None,...]
        if self.scaler_aia is not None:
            ch_dataset = self.scaler_aia.transform(ch_dataset)
            ar_dataset = self.scaler_aia.transform(ar_dataset)
        # Should return a torch tensor of form [1,512,34,channels]
        
        ch_net_area = np.sum(ch_segmentation)/(ch_segmentation.shape[0]*ch_segmentation.shape[1])  # Calculate area of open-field-line regions    
        # Calculate the net flux per measurement per passband
        ch_net_fluxes = np.zeros(ch_dataset.shape[3])
        for passband in range(ch_dataset.shape[3]):
            ch_net_fluxes[passband] = np.sum(ch_dataset[0, :, :, passband])
        ch_data = np.append(ch_net_fluxes, ch_net_area)

        ar_net_area = np.sum(ar_segmentation)/(ar_segmentation.shape[0]*ar_segmentation.shape[1])  # Calculate area of closed-field-line regions    
        # Calculate the net flux per measurement per passband
        ar_net_fluxes = np.zeros(ar_dataset.shape[3])
        for passband in range(ar_dataset.shape[3]):
            ar_net_fluxes[passband] = np.sum(ar_dataset[0, :, :, passband])
        ar_data = np.append(ar_net_fluxes, ar_net_area)
        
        in_data = np.append(ch_data, ar_data)
                
        return in_data
    