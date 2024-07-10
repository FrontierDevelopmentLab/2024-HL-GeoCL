import datetime as dt
import pickle
import pdb

import numpy as np
import pandas as pd
import xgboost as xgb
from SHEATH_atomic_predict import AtomicSamurai, CloudFetcher
from preprocessings import Preprocessor_CH
from tqdm import tqdm

# -------
# Define the cloud fetcher object to get the data
cloudfetcher_object = CloudFetcher(
    zarr_bucket="us-fdlx-ard-zarr-synoptic",
    aia_path="fdl-sdoml-v2/sdomlv2.zarr",
    hmi_path="fdl-sdoml-v2/sdomlv2_hmi.zarr",
)

# Define the scaler
scaler_aia = pickle.load(open("../logs/scaler_X.scaler", "rb"))
scaler_omni = pickle.load(open("../logs/scaler_y.scaler", "rb"))

# Define the preprocessing module
preprocessor = Preprocessor_CH(npix=17, n_size=512, ch_mask=True, scaler_aia=scaler_aia)

# Define the NN model
HSE_model = xgb.Booster({"nthread": -1})  # init model
HSE_model.load_model("../logs/SHEATH_xgb.model")  # load data

# Load the solar wind variable names and scalers
sw_vars_list = pickle.load(open("../logs/sw_variables.pickle", "rb"))

# Instantiate the atomic inference module
atom = AtomicSamurai(cloudfetcher_object, HSE_model)

image_indices = pd.read_csv("aligndata_2010_2020_AIA_HMI.csv")
image_indices.sort_values(by="Time", inplace=True)
image_indices.set_index("Time", inplace=True, drop=True)
image_indices.index = pd.to_datetime(image_indices.index)
stime = dt.datetime.strptime("2017-09-26 00:00:00", "%Y-%m-%d %H:%M:%S")
etime = dt.datetime.strptime("2017-09-30 00:00:00", "%Y-%m-%d %H:%M:%S")
start_index = np.argmin(np.abs(stime - image_indices.index))
end_index = np.argmin(np.abs(etime - image_indices.index))
image_indices = image_indices.iloc[start_index:end_index]

predictions = []
for timestamp in tqdm(image_indices.index, desc="Performing inference"):
    try:
        idxs = image_indices.loc[timestamp]
    except Exception:
        pdb.set_trace()

    sw_values = atom.atomic_inference(
        aia_timestamp=timestamp,
        aia_idx=idxs[:-1],
        hmi_timestamp=timestamp,
        hmi_idx=idxs[-1],
        scaler_aia=scaler_aia,
        scaler_omni=scaler_omni,
    )
    predictions.append(pd.DataFrame(sw_values))

predictions = pd.concat(predictions, axis=0)

predictions = scaler_omni.inverse_transform(predictions)

predictions = pd.DataFrame(
    predictions, columns=["bx", "by", "bz", "vx", "number_density", "temperature"]
)
predictions["Time"] = image_indices.index
predictions["vz"] = np.zeros(len(predictions))  # Assume velocity is purely radial
predictions["vy"] = np.zeros(len(predictions))
predictions["xgse"] = np.zeros(len(predictions))
predictions["ygse"] = np.zeros(len(predictions))
predictions["zgse"] = np.zeros(len(predictions))
predictions["clock_angle"] = np.arctan(predictions["by"] / predictions["bz"])
predictions["density"] = (
    predictions["number_density"] * 1e6 * 1.6726e-27
)  # Assuming Hydrogen solar wind
predictions["psw"] = (predictions["vx"] * 1000) ** 2 * predictions[
    "density"
]  # Ram pressure in radial, Hydrogen solar wind

predictions.drop(columns=["number_density"], inplace=True)
predictions.sort_values(by="Time", inplace=True)
predictions.set_index("Time", inplace=True, drop=True)
predictions.index = predictions.index.floor(freq="min")

# Ensure columns are in the right order
predictions = predictions[
    [
        "bx",
        "by",
        "bz",
        "vx",
        "vy",
        "vz",
        "density",
        "psw",
        "temperature",
        "xgse",
        "ygse",
        "zgse",
        "clock_angle",
    ]
]

predictions.to_hdf(
    "../data_local/omni/sheath_sw_data_test_withy.h5", key=str(stime.year)
)
print("Inference complete!")
