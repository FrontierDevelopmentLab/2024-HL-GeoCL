import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from astropy.constants import iau2012 as const
import astropy.units as u

def backtrace_radial(vel):
    """
        Ballistically backtrace a plasma parcel assuming radial flow.
        vel: np.ndarray or scalar in km/s.
    """
    return const.au.to('km').value/(vel*3600*24.0)
def get_backtrace_date(vel,sw_date):
    time = backtrace_radial(vel)
    return (sw_date-pd.to_timedelta(time,unit='day'))

def backtrace_spiral(v,theta):
    """
        Ballistically backtrace a plasma parcel assuming spiral flow.
        vel: np.ndarray or scalar in km/s.
    """
    omega_sun = np.deg2rad(360.0)/(27.2*24*3600)
    ro = const.R_sun.to('km').value
    au = const.au.to('km').value
    theta = np.deg2rad(theta)
    arg = omega_sun*(au-ro)*np.sin(theta)/v
    return np.arcsin(arg)/(omega_sun*np.sin(theta)*(3600*24.0))

def backtrace_spiral_date(vel,sw_date):
    """
        Ballistically backtrace a plasma parcel assuming spiral flow.
        vel: np.ndarray or scalar in km/s.
    """
    time = backtrace_spiral(vel,90)
    return (sw_date-pd.to_timedelta(time,unit='day'))


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