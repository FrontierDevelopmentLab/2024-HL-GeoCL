import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inference.SHEATH_atomic_predict import CloudFetcher
from tqdm import tqdm

image_indices = pd.read_csv("aligndata_2010_2020_AIA_HMI.csv")
image_indices.sort_values(by="Time", inplace=True)
image_indices.set_index("Time", inplace=True, drop=True)
image_indices.index = pd.to_datetime(image_indices.index)
stime = dt.datetime.strptime("2017-09-26 00:00:00", "%Y-%m-%d %H:%M:%S")
etime = dt.datetime.strptime("2017-09-30 00:00:00", "%Y-%m-%d %H:%M:%S")
start_index = np.argmin(np.abs(stime - image_indices.index))
end_index = np.argmin(np.abs(etime - image_indices.index))
image_indices = image_indices.iloc[start_index:end_index]

cloud_fetcher = CloudFetcher(
    zarr_bucket="us-fdlx-ard-zarr-synoptic",
    aia_path="fdl-sdoml-v2/sdomlv2.zarr",
    hmi_path="fdl-sdoml-v2/sdomlv2_hmi.zarr",
)

for i, timestamp in tqdm(enumerate(image_indices.index[::5])):
    aia_image = cloud_fetcher.load_aia_image(
        time=timestamp, idxs=image_indices.loc[timestamp]
    )["193A"]
    hmi_image = cloud_fetcher.load_hmi_image(
        time=timestamp, idx=image_indices.loc[timestamp]["idx_Bx"]
    )["Bz"]
    plt.figure()
    plt.axis("off")
    plt.imshow(np.log10(aia_image))
    plt.title(timestamp.strftime("%Y-%m-%d %X"), fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sheath_data/193A_movie/193A_{i}.png")
    plt.figure()
    plt.axis("off")
    plt.imshow(np.log10(hmi_image))
    plt.title(timestamp.strftime("%Y-%m-%d %X"), fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sheath_data/Bz_movie/Bz_{i}.png")
