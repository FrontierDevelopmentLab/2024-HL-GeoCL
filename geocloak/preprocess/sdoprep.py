"""
This python module is the part of preprocess libarary. Which preprocess
the SDO data from SDOMLv2 and generate the featuire vector
returned as python dictionary.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM


class SDODataPreprocess:
    """
    This class takes an dictionary with 12 files from SDOMLv2, which containes 10 chnnels of AIA
    and 3 magnetic field components of the HMI vector data. Use the Gaussian Mixture Model
    to identify the Coronal Holes(CH) and Active Region (AR) mask and then use these mask
    to calculate the list of paramters which are the input vector for SHEATH.
    """

    def __init__(self, sdofiles, timestamp, outpath="sdo_processed_data", crop=True):
        self.sdofiles = sdofiles
        self.timestamp = timestamp
        self.outpath = outpath
        self.crop = crop

        # These parameters need to be adjusted accoring to the image.
        self.npixel = 512
        self.lonwidth = 17
        self.n_comp = 3

        # Estimate the radius of solar disk based
        # on size and pixel scale of the image
        self.radius = (695700.0 / 725.0) / (0.6 * (4096 / 512.0))  # very approximate

    def __get_diskmask(self):
        """
        Genearte solar disk mask based on the attribute of the class.

        Returns
        -------
        mask: np.ndarray
            The binary mask contatining the information about only the disk region
            inside +/- lonwidth pixel from the center of the image.
        """
        # Define the central pixels
        center = np.array([self.npixel / 2.0, self.npixel / 2.0])
        xg = np.arange(0, self.npixel)
        xgrid, ygrid = np.meshgrid(xg, xg)
        distance = ((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2).astype(int)

        # Disc mask
        mask = np.sign(distance - self.radius**2)
        mask[mask > 0] = np.nan
        mask = np.abs(mask)
        mask = mask[
            :, self.npixel // 2 - self.lonwidth : self.npixel // 2 + self.lonwidth
        ]
        return mask

    def get_193mask(self, region=["CH", "AR"], n_comp=3):
        """
        This function takes the 193A image and generate the mask for the
        Coronal Holes (CH) and Active Regions (AR) using the Gaussian Mixture Model.
        The function first generate the solar disk mask and then apply the GMM on
        the image data inside the disk region. The function then return the mask
        for the CH and AR.


        Parameters
        ----------
        im : np.ndarray
            The input image array of AIA channel 193A.
        region : list, optional
            Mangnetic features need to included, by default ["CH", "AR"]
        n_comp : int, optional
            Number of component used in GMM, by default 3

        Returns
        -------
        segment: dict
            Segmented image mask, containg mask of CH and AR.
        """

        # Check for input for magnetic regions.
        if not all([True if x in ["AR", "CH", "QS"] else False for x in region]):
            return None

        im = self.sdofiles["193A"]
        im[im < 1] = 1.0
        im = np.log10(im)

        # Get the mask of the image
        mask = self.__get_diskmask()
        # mask[np.isnan(mask)] = 0

        # Crop the image abouth 17 pixels in longitude
        if not self.crop:
            og = im[
                :, self.npixel // 2 - self.lonwidth : self.npixel // 2 + self.lonwidth
            ]
        else:
            og = im

        og[np.isnan(og)] = 0.0

        # Crop the image abouth 17 pixels in longitude by multiplying with mask
        sample = og * mask
        sample = sample[~np.isnan(sample)]

        # Check for number of pixels in the masked image
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

        # Identify segments for the data
        segments = {}
        for k in region:
            tmp = (
                th_gmm
                == np.where(
                    np.asarray(gmodel.means_) == centroidfnlist[k](gmodel.means_)
                )[0]
            )
            segments[k] = np.reshape(tmp, list(og.shape))
        return segments

    def get_features(self):
        """
        This function takes the dictionary of SDO files and calculate the
        feature vector for the SHEATH model.

        Returns
        -------
        feature_vector : list
            The list of feature vector for the SHEATH model.
        """
        _mask = self.get_193mask()
        newmask = self.__get_diskmask()
        newmask[np.isnan(newmask)] = 0

        chmask = _mask["CH"] * newmask
        armask = _mask["AR"] * newmask
        feature_vector = {"CHArea": np.nanmean(chmask), "ARArea": np.nanmean(armask)}
        for im in self.sdofiles:
            if not self.crop:
                _imagedata = self.sdofiles[im][
                    :,
                    self.npixel // 2 - self.lonwidth : self.npixel // 2 + self.lonwidth,
                ]
            else:
                _imagedata = self.sdofiles[im]
            feature_vector[f"CH{im}"] = np.nanmean(_imagedata * chmask)
            feature_vector[f"AR{im}"] = np.nanmean(_imagedata * armask)
        return feature_vector

    def feature_vector(self):
        """
        This function takes the dictionary of SDO files and calculate the
        feature vector for the SHEATH model.

        Returns
        -------
        feature_vector : dict
            The dictionary of feature vector along with the time stamp for the SHEATH model.
        """
        _timestamp = self.timestamp
        if isinstance(_timestamp, str):
            if _timestamp.endswith("Z"):
                _timestamp = pd.to_datetime(_timestamp)
            elif _timestamp.endswith("_TAI"):
                _timestamp = pd.to_datetime(
                    datetime.strptime(_timestamp, "%Y.%m.%d_%H:%M:%S_TAI")
                )
            else:
                try:
                    _timestamp = pd.to_datetime(_timestamp)
                except Exception:
                    raise SyntaxError("Unable to interprate time format.")
        elif isinstance(_timestamp, datetime):
            _timestamp = pd.to_datetime(_timestamp)
        feature_vector = {
            "Time": _timestamp.round("min").isoformat()
        } | self.get_features()
        return feature_vector
