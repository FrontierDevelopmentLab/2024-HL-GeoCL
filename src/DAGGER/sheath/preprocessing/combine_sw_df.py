import numpy as np
import pandas as pd 
import datetime
from astropy.constants import iau2012 as const

start = 2020
end = 2023
d1 = pd.read_hdf(f"omni_preprocess_{start}_{end}.h5")

start = 2020
end = 2023
d2 = pd.read_hdf(f"omni_preprocess_{start}_{end}.h5")

start = 2020
end = 2023
d3 = pd.read_hdf(f"omni_preprocess_{start}_{end}.h5")

Data = pd.concat([d1,d2,d3])

print("Columns in the OMNI file: ")
print(repr(Data.columns))

print("Max values of different variables")
_ = [print(np.nanmax(Data[v].values)) for v in Data.columns]

Data = Data.dropna()
Data.to_hdf(f"omni_preprocess_complete.h5",key="omni",mode="w")