import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys
sys.path.append("../")
from utils.preprocessing import get_backtrace_date,backtrace_spiral_date


"""
    For the original OMNI dataset, we backtrace each time stamp back to AIA data. This is BEFORE TRAIN-TEST-VAL split.
"""

omni_path = "/home/jupyter/Vishal/omni/omni_preprocess_1hour_full.h5"
omni_data = pd.read_hdf(omni_path)
dates = omni_data.Date.values
speeds = omni_data['Speed, km/s'].values

sun_dates = get_backtrace_date(speeds,dates)
print(f"{dates[0]}: {speeds[0]} -> {sun_dates[0]}")
print(f"{dates[-1]}: {speeds[-1]} -> {sun_dates[-1]}")
np.save("../logs/sun_backtraced_dates.npy",sun_dates)