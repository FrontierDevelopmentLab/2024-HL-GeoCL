import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from train_SHEATH import *


# Loading the placeholder for ARD's ARD (which is just our train/test set for now)
in_dataset = np.asarray([np.load(v) for v in sorted(glob(f"{DATAPATH}masked*.npy"))]).transpose([1,2,3,0])


class IrradianceInferenceModel:
    def __init__(self):
        self.zarr_bucket = "us-fdlx-landing"
        self.aia_path = "fdl-sdoml-v2/sdomlv2_small.zarr"
        self.hmi_path = "fdl-sdoml-v2/sdomlv2_hmi_small.zarr"
        
        self.aia_wavelengths = ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"]
        self.hmi_components = ["Bx", "By", "Bz"]
        

    def predict(self, aia_image, forward_passes=0):
        aia_image = self.prepare_aia(aia_image)
        with torch.no_grad():
            pred_irradiance = self.model.forward_unnormalize(aia_image).numpy()
        
        pred_irradiance = pd.Series({ion: pred_irradiance[0][i] for i, ion in enumerate(self.eve_ions)})
        return pred_irradiance


    def prepare_aia(self, aia_image):
        for wavelength in self.aia_wavelengths:
            aia_image[wavelength] -= self.aia_normalizations[wavelength]["mean"]
            aia_image[wavelength] /= self.aia_normalizations[wavelength]["std"]

        aia_image = np.array([np.stack([aia_image[wavelength] for wavelength in self.aia_wavelengths], axis=0)])
        aia_image = torch.from_numpy(aia_image)
        return aia_image


    def load_aia_image(self, time="2010-08-17T23:54:11.07Z", idx=6134):
        aia_image = {}
        for wavelength in self.aia_wavelengths:
            # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            year = int(time.year)
            aia_image[wavelength] = self.aia_data[year][wavelength][idx,:,:]
        return aia_image
    
    
     def load_hmi_image(self, time, idx):
        hmi_image = {}
        for component in self.hmi_components:
            # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            year = int(time.year)
            hmi_image[component] = self.aia_data[year][component][idx,:,:]
        return hmi_image
        
    
    def get_zarr_root(self) -> zarr.hierarchy.Group:
        print(f"Connecting to zarr root in path: {self.aia_path}, {self.hmi_path}, and bucket: {self.zarr_bucket}")

        gcp_zarr = gcsfs.GCSFileSystem(project='us-fdl-x', bucket=self.zarr_bucket, access="read_only", requester_pays=True)
        store_aia = gcsfs.GCSMap(root=f"{self.zarr_bucket}/{self.aia_path}", gcs=gcp_zarr, check=False, create=True)
        store_hmi = gcsfs.GCSMap(root=f"{self.zarr_bucket}/{self.hmi_path}", gcs=gcp_zarr, check=False, create=True)

        aia_root = zarr.group(store=store_aia)
        hmi_root = zarr.group(store=store_hmi)

        return aia_root, hmi_root


    def atomic_infer(self, sdo_timestamp, sdo_idx, ):
      """
      """
        # Timestamp and (index=-1)
        # Access zarr bucket

        # Get desired channels and check alignment and channel order
        # Crop
        # (Segmentation?) 
        # Scaling
        # model.predict()
        # return listlike of 7 OMNI features, the associated OMNI timestamp, and the associated SDO timestamp
        # Later: push to bucket or save to file locally