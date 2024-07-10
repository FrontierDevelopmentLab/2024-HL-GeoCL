import pandas as pd
from astropy.time import Time


# HMI time is in TAI, not UTC. So convert it first.
def stringchange(string):
    new_string = string.replace("_TAI", "Z")
    new_string = new_string.replace("_", "T")
    new_string = new_string.replace(".", "-")
    new_string = new_string.replace("Z", ".00")
    return new_string


def convert_hmi_time_utc(time_hmi):
    t_obs_new = [stringchange(string) for string in time_hmi]
    t = Time(t_obs_new, format="isot", scale="tai")
    t_obs_new = pd.to_datetime(t.utc.value, utc=True)
    return t_obs_new
