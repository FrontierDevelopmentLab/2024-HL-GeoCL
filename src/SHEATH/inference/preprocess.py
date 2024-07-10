import numpy as np
from sklearn.mixture import GaussianMixture as GMM


def get_mask(n_size=512):
    # Generate mask to get only the disc. I don't care about the limb
    # Define solar disc
    center = np.array([256, 256])
    radius = (695700.0 / 725.0) / (0.6 * (4096 / n_size))  # very approximate
    xg = np.arange(0, n_size)
    xgrid, ygrid = np.meshgrid(xg, xg)
    distance = ((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2).astype(int)

    # disc mask
    mask = np.sign(distance - radius**2)
    mask[mask > 0] = np.nan
    mask = np.abs(mask)
    return mask


def GetMorphologicalStructure(og, mask, region=["AR"], n_comp=3):
    """
    This function segments out the active regions, coronal holes and quiet sun from our images. It uses a Gaussian Mixture Model (GMM)
    to segement out the regions. GMM can be understood to be a generalization of Otsu thresholding.
    This function is a generalization of `GetActiveRegions`, where:
        1. Minimum of centroid mean corresponds to CHs.
        2. Maximum of centroid mean corresponds to ARs.
        3. Meidan of centroid mean corresponds to QS.
    Inputs:
        sample: img of shape [isize,isize], minvalue = 0 and maxvalue = 1

    This function is a part of suitpy package.
    """
    # Initial smoothing
    # sample = cv2.bilateralFilter(og.astype(np.float32),9,75,75)
    sample = og * mask
    sample = sample[~np.isnan(sample)]

    if len(np.unique(sample)) <= 3:
        segments = {}
        for k in region:
            segments[k] = np.zeros_like(og)
        return segments

    # Define the mixture model, and take the component with highest mean value.
    gmodel = GMM(n_components=n_comp)
    gmodel.fit(np.reshape(sample, [-1, 1]))
    th_gmm = gmodel.predict(np.reshape(og, [-1, 1]))
    centroidfnlist = {"AR": np.max, "CH": np.min, "QS": np.median}
    assert all([True if x in ["AR", "CH", "QS"] else False for x in region])
    segments = {}
    mask
    for k in region:
        tmp = (
            th_gmm
            == np.where(np.asarray(gmodel.means_) == centroidfnlist[k](gmodel.means_))[
                0
            ]
        )
        segments[k] = np.reshape(tmp, list(og.shape))
    return segments


def subsample_and_segment(og, mask, region=["CH"], ncomp=3):
    segmentation = GetMorphologicalStructure(og, mask, region=region, n_comp=ncomp)
    return segmentation["CH"], segmentation["AR"]


class Preprocessor_CH_xgb:
    def __init__(self, npix=17, n_size=512, ch_mask=True, scaler_aia=None):
        self.ch_mask = ch_mask
        self.scaler_aia = scaler_aia
        self.npix = npix
        self.n_size = n_size

    def preprocess(
        self, aia_data, hmi_data, aia_wavelengths, hmi_components, scaler_aia
    ):
        ind_193 = aia_wavelengths.index("193A")
        n_size = self.n_size
        HALF = int(n_size // 2)
        npix = self.npix
        mask = get_mask(n_size=n_size)[:, HALF - npix : HALF + npix]
        if self.ch_mask:
            ch, ar = subsample_and_segment(
                aia_data[ind_193][:, HALF - npix : HALF + npix],
                mask,
                region=["CH", "AR"],
                ncomp=3,
            )
        else:
            ch = np.ones_like(aia_data[ind_193][:, HALF - npix : HALF + npix])
            ar = np.ones_like(aia_data[ind_193][:, HALF - npix : HALF + npix])

        feature_array = [ch, ar]
        for channel in np.arange(len(aia_data)):
            feature_array.append(aia_data[channel][:, HALF - npix : HALF + npix] * ch)
            feature_array.append(aia_data[channel][:, HALF - npix : HALF + npix] * ar)
        for channel in np.arange(len(hmi_data)):
            feature_array.append(hmi_data[channel][:, HALF - npix : HALF + npix] * ch)
            feature_array.append(hmi_data[channel][:, HALF - npix : HALF + npix] * ar)
        feature_array = np.asarray(feature_array)
        feature_array = np.nanmean(feature_array, axis=(-1, -2)).reshape([1, -1])
        feature_array = self.scaler_aia.transform(feature_array)
        return feature_array
