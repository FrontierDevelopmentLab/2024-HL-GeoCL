"""
Fitting B time series (animation)
---------------------------------

This example demonstrates how to fit generic B observation inputs and fit an SECS system
to make predictions on a separate grid and compare the results.
"""
from matplotlib.animation import FuncAnimation
import numpy as np
from sec import SECS


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


R_earth = 6371e3  # in meters


# specify the SECS grid
lat, lon, r = np.meshgrid(np.linspace(0, 20, 30),
                          np.linspace(0, 20, 30),
                          R_earth, indexing='ij')

secs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                            lon.reshape(-1, 1),
                            r.reshape(-1, 1)))

# Set up the class
secs = SECS(sec_df_loc=secs_lat_lon_r)

# Make a grid of input observations spanning
# (-10, 10) in latitutde and longitude
lat, lon, r = np.meshgrid(np.linspace(1, 10, 11),
                          np.linspace(1, 10, 11),
                          R_earth, indexing='ij')
obs_lat = lat[..., 0]
obs_lon = lon[..., 0]
obs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                           lon.reshape(-1, 1),
                           r.reshape(-1, 1)))
nobs = len(obs_lat_lon_r)

# Create the synthetic magnetic field data as a function
# of time
ts = np.linspace(0, 2*np.pi)[:2]
bx = np.random.rand(len(ts))
by = np.random.rand(len(ts))
bz = np.random.rand(len(ts))
# ntimes x 3
B_obs = np.column_stack([bx, by, bz])
ntimes = len(B_obs)

# Repeat that for each observatory
# ntimes x nobs x 3
B_obs = np.repeat(B_obs[:, np.newaxis, :], nobs, axis=1)
# Make it more interesting and add a sin wave in spatial
# coordinates too
B_obs[:, :, 0] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 0]))
B_obs[:, :, 1] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 1]))

B_std = np.ones(B_obs.shape)
# Ignore the Z component
B_std[..., 2] = np.inf
# Can modify the standard error as a function of time to
# see how that changes the fits too
# B_std[:, 0, 1] = 1 + ts

# Fit the data, requires observation locations and data
secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, obs_std=B_std, epsilon=0.01)

# Create prediction points
# Extend it a little beyond the observation points (-11, 11)
lat, lon, r = np.meshgrid(np.linspace(1, 10, 11),
                          np.linspace(1, 10, 11),
                          R_earth, indexing='ij')
pred_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                            lon.reshape(-1, 1),
                            r.reshape(-1, 1)))

# Call the prediction function
B_pred = secs.predict(pred_lat_lon_r)

# Now set up the plots
fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True)
cmap = plt.get_cmap('RdBu_r')
t = 0

ax[0].pcolormesh(lon[..., 0], lat[..., 0], B_pred[t, :, 0].reshape(obs_lon.shape))
ax[2].pcolormesh(lon[..., 0], lat[..., 0], B_pred[t, :, 1].reshape(obs_lon.shape))
ax[1].pcolormesh(obs_lon, obs_lat, B_obs[t, :, 0].reshape(obs_lon.shape))
ax[3].pcolormesh(obs_lon, obs_lat, B_obs[t, :, 1].reshape(obs_lon.shape))

ax[0].set_ylabel("Observations")
ax[0].set_title("B$_{X}$")
ax[2].set_title("B$_{Y}$")
plt.show()