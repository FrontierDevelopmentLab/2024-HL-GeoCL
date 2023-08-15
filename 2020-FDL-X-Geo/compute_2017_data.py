from utils.data_utils import get_iaga_data, get_input_data, load_cached_data,get_wiemer_data,get_iaga_data_as_list, get_iaga_max_stations
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from utils.splitter import get_sequences

dataset_2015 = dict(np.load('2015_supermag_omni_data.npz'))
dataset = {}

dataset["stations"] = dataset_2015["stations"]
dates,data,features,reg = get_iaga_data_as_list(base="data_local/iaga/",year=[2017])

start_time = datetime.strptime('26.09.2017 00:00:00', '%d.%m.%Y %H:%M:%S').timestamp()
end_time = datetime.strptime('30.09.2017 00:00:00', '%d.%m.%Y %H:%M:%S').timestamp()

indices = np.where(np.logical_and(dates>=start_time, dates<=end_time))
indices = indices[0]

dataset["dates"] = dates[indices]
dataset["data"] = data[indices]
dataset["features"] = features
dataset["reg"] = reg[indices]

df = pd.DataFrame()
df["seconds"] = dataset["dates"]
df['index'] = range(len(df))

dataset["idx"] = get_sequences(df, 120, 30)

extra_input_features = []
df = get_input_data(omni_path="data_local/omni/sheath_sw_data.h5",indices_path="data_local/supermag_indices/",
               indices_to_use=extra_input_features, year=[2017])

dataset["omni_features"] = df.columns
dataset["omni"] = df.values

np.savez("2017_supermag_omni_data.npz", **dataset)