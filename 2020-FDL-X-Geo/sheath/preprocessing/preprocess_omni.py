import numpy as np
import pandas as pd 
import datetime
from astropy.constants import iau2012 as const

"""
    Script to preprocess OMNI data. 
    1. OMNI data has dates in format YEAR - No.of days from Jan 1 - Hour - Minutes. This must be converted to a correct datetime variable.
    2. OMNI data contains missing values as 9999.0, 999.9 and so on. These must be replaced with actual np.nan.

"""

data_path = "omni2_data.lst"
fmt_path = "omni2_format.fmt"

#Read OMNI formatter
tmp = pd.read_fwf(fmt_path,sep=" ",engine='python',header=1).values[:,0]
heads = [" ".join(t.split(" ")[1:]) for t in tmp]
#Read data file and assign correct column names
Data = pd.read_csv(data_path,sep="\s{1,}",engine='python',header=None,names=heads)
#Convert YR-DOY-HR to correct datetime.
Data['Date'] = Data.apply(lambda row: datetime.datetime(int(row.Year),1,1,int(row.Hour),int(row.Minute))+datetime.timedelta(row.Day), axis=1)

Data = Data.drop(["Year","Day","Hour","Minute"],axis=1)
Data = Data[ ['Date'] + [ col for col in Data.columns if col != 'Date' ] ]
#Convert all the 9999s to nan
Data.replace({9999.0:np.nan,999.9:np.nan,9999999.0:np.nan,9.999:np.nan,999.99:np.nan,9999.99:np.nan},inplace=True)


print("Columns in the OMNI file: ")
print(repr(Data.columns))

print("Max values of different variables")
_ = [print(np.nanmax(Data[v].values)) for v in Data.columns]
Data.to_hdf("omni_preprocess.h5",key="omni",mode="w")