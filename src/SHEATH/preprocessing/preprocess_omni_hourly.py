import numpy as np
import pandas as pd
import pdb

"""

    It is safest to start having OMNI data from 2010-06-01. We dont have data before that!!
    Also, end on 2020

"""

data_path = "omni2_1hour_full.lst"
fmt_path = "omni2_1hour_full.fmt"

# Read OMNI formatter
tmp = pd.read_fwf(fmt_path, sep=" ", engine="python", header=1).values[:, 0]
heads = [" ".join(t.split(" ")[1:]) for t in tmp]
# Read data file and assign correct column names
Data = pd.read_csv(data_path, sep="\s{1,}", engine="python", header=None, names=heads)
years = [pd.Timestamp(a, 1, 1) for a in Data["Year"].values]
times = [
    pd.to_datetime(ia, unit="D", origin=iv) for ia, iv in zip(Data["DOY"].values, years)
]
final_datetime = [
    t + pd.to_timedelta(h, unit="h") for t, h in zip(times, Data["Hour"].values)
]

# Conver YR-DOY-HR to correct times.
Data["Date"] = pd.to_datetime(final_datetime)
Data = Data.drop(["Year", "DOY", "Hour"], axis=1)
Data = Data[["Date"] + [col for col in Data.columns if col != "Date"]]
# Convert all the 9999s to nan
Data.replace(
    {
        9999.0: np.nan,
        999.9: np.nan,
        9999999.0: np.nan,
        9.999: np.nan,
        999.99: np.nan,
        9999.99: np.nan,
        99999.9: np.nan,
    },
    inplace=True,
)

pdb.set_trace()

print("Columns in the OMNI file: ")
print(repr(Data.columns))

print("Max values of different variables")
_ = [print(np.nanmax(Data[v].values)) for v in Data.columns]
# Data.to_hdf(f"omni_preprocess_{start}_{end}.h5",key="omni",mode="w")
Data = Data.dropna()

start = pd.to_datetime("2010-06-01")
end = pd.to_datetime("2020-12-31")
mask = (Data["Date"] > start) & (Data["Date"] <= end)
Data = Data.loc[mask]

Data.to_hdf("omni_preprocess_1hour_full.h5", key="omni", mode="w")
