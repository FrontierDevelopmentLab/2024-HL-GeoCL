import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import zarr
import gcsfs

class CloudFetcher:
    def __init__(self,zarr_bucket="us-fdlx-landing", aia_path = "fdl-sdoml-v2/sdomlv2_small.zarr",
                 hmi_path = "fdl-sdoml-v2/sdomlv2_hmi_small.zarr",
                 aia_wavelengths = ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"],
                 hmi_components = ["Bx", "By", "Bz"]):
        """
            Define the zarr buckets and aia, hmi path.
        """
        self.zarr_bucket = zarr_bucket
        self.aia_path = aia_path
        self.hmi_path = hmi_path
        
        self.aia_wavelengths = aia_wavelengths
        self.hmi_components = hmi_components
        # Store the AIA and HMI root objects
        self.aia_data,self.hmi_data = self.get_zarr_root()
    
    def get_zarr_root(self) -> zarr.hierarchy.Group:
        """
            Function gets a pointer to the aia and hmi repositories in Google cloud.
        """
        print(f"Connecting to zarr root in path: {self.aia_path}, {self.hmi_path}, and bucket: {self.zarr_bucket}")

        gcp_zarr = gcsfs.GCSFileSystem(project='us-fdl-x', bucket=self.zarr_bucket, access="read_only", requester_pays=True)
        store_aia = gcsfs.GCSMap(root=f"{self.zarr_bucket}/{self.aia_path}", gcs=gcp_zarr, check=False, create=True)
        store_hmi = gcsfs.GCSMap(root=f"{self.zarr_bucket}/{self.hmi_path}", gcs=gcp_zarr, check=False, create=True)

        aia_root = zarr.group(store=store_aia)
        hmi_root = zarr.group(store=store_hmi)
        return aia_root, hmi_root
    def load_aia_image(self, time, idx):
        """
            Given a particular datetimestamp, get the nearest AIA images.
            TODO: Find the nearest time and use it. Discard idx.
        """
        aia_image = {}
        if isinstance(time,str):
            time = pd.to_datetime(time)
        for wavelength in self.aia_wavelengths:
            # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            year = int(time.year)
            aia_image[wavelength] = self.aia_data[year][wavelength][idx,:,:]
        return aia_image
    def load_hmi_image(self, time, idx):
        """
            Given a particular datetimestamp, get the nearest HMI images.
            TODO: Find the nearest time and use it. Discard idx.
        """
        hmi_image = {}
        if isinstance(time,str):
            time = pd.to_datetime(time)
        for component in self.hmi_components:
            # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            year = int(time.year)
            hmi_image[component] = self.hmi_data[year][component][idx,:,:]
        return hmi_image

class AtomicSamurai:
    """
        Atomic Samurai takes in a CloudFetcher object, a preprocessing function, and the SHEATH module, to produce
        inferences. The idea is to perform inference for a single timestamp using this module, and then run it for
        multiple timestamps.
        
        The SHEATH module will need to have an inbuilt dataloader and inference model. This is done since 
        we may choose to use any DL framework for inference or may stick to CPUs too. The side facing the data
        does not care about any framework, and any inference part should come as a part of the module being used. 
        
        Refernce: Atomic Samurai is an S-class hero in One Punch man who fights with a Katana [you see where I am going with this].
    """
    def __init__(self,cloudfetcher_object, sheathmodule_object):
        self.cloudfetcher_object = cloudfetcher_object
        self.sheathmodule_object = sheathmodule_object


    def atomic_inference(self, aia_timestamp, aia_idx, hmi_timestamp, hmi_idx):
        """
              This function takens in the AIA and HMI timestamps and the corresponding indices to perform inference.
              This calls the cloudfetcher object to get the data, apply the preprocessing, and pass in the array to sheat module.
        """
        aia_data = self.cloudfetcher_object.load_aia_image(aia_timestamp,aia_idx)
        hmi_data = self.cloudfetcher_object.load_hmi_image(hmi_timestamp,hmi_idx)
        input_datacube = self.sheathmodule_object.preprocessor.preprocess(aia_data,hmi_data,
                                                             self.cloudfetcher_object.aia_wavelengths,
                                                             self.cloudfetcher_object.hmi_components)
        results = self.sheathmodule_object.predict(input_datacube)
        return results
