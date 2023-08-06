import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import cv2
import sys
from utils.torch_utils import _float

def get_mask(n_size = 512.0):
    # Generate mask to get only the disc. I don't care about the limb
    # Define solar disc
    center = np.array([256,256])
    radius = (695700.0/725.0)/(0.6*(4096/n_size)) #very approximate
    print(f"Solar disc radius : {radius}")
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
        import pdb; pdb.set_trace()
        
        if self.ch_mask:
            segmentation = subsample_and_segment(aia_data[ind_193,:,int(n_size//2)-npix:int(n_size//2)+npix], mask,
                                                 region = ['CH'], ncomp = 3)
            segmentation = segmentation["CH"]
        else:
            segmentation = np.ones_like(aia_data[ind_193,:,int(n_size//2)-npix:int(n_size//2)+npix])

        aia_data = aia_data[:,:,int(n_size//2)-npix:int(n_size//2)+npix]*(segmentation[None,...])
        hmi_data = hmi_data[:,:,int(n_size//2)-npix:int(n_size//2)+npix]*(segmentation[None,...])
        # The input data should be of the form [512,34,12]
        in_data = np.concatenate([aia_data,hmi_data]).transpose([1,2,0])[None,...]
        if self.scaler_aia is not None:
            in_data = self.scaler_aia.transform(in_data)
        # Should return a torch tensor of form [None,512,34,channel]
        return _float(in_data)