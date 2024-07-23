"""
Fitting B time series (animation)
---------------------------------

This example demonstrates how to fit generic B observation inputs and fit an SECS system
to make predictions on a separate grid and compare the results.
"""
import numpy as np
from sec import T_df
from spherical_harmonics import ridge_regression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



R_earth = 6371  # in km
R_ionosphere = R_earth + 100  # in km

# specify the secs grid
lat, lon, r = np.meshgrid(np.linspace(0, 20, 30),  # in deg [-90, 90]
                          np.linspace(0, 20, 30),  # in deg [0, 360]
                          R_ionosphere,  # in km
                          indexing='ij')

secs_lat_lon_r = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1), r.reshape(-1, 1)))

# Make a grid of input observations spanning
# (-10, 10) in latitude and longitude
lat, lon, r = np.meshgrid(np.linspace(1, 10, 11),  # in deg [-90, 90]
                          np.linspace(1, 10, 11),  # in deg [0, 360]
                          R_earth,  # in km
                          indexing='ij')

obs_lat_lon_r = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1), r.reshape(-1, 1)))

# create synthetic data
B_obs = np.random.rand(len(obs_lat_lon_r), 3)
B_obs[:, 0] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 0]))
B_obs[:, 1] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 1]))
B_obs[:, 2] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 2]))

# get T matrix
T_mat = T_df(obs_loc=obs_lat_lon_r, sec_loc=secs_lat_lon_r)
I_sol = ridge_regression(basis_matrix=T_mat,
                         data=np.ndarray.flatten(B_obs),
                         lambda_=0.01)

# create prediction points
# Extend it a little beyond the observation points (-11, 11)
n_lat = 10
n_lon = 20
lat, lon, r = np.meshgrid(np.linspace(-10, 10, n_lat),
                          np.linspace(0, 30, n_lon),
                          R_earth,
                          indexing='ij')

pred_lat_lon_r = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1), r.reshape(-1, 1)))

# predict
B_pred = T_df(obs_loc=pred_lat_lon_r, sec_loc=secs_lat_lon_r) @ I_sol

# plot
fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True)

ax[0].pcolormesh(lon[..., 0], lat[..., 0], B_pred.reshape((n_lat, n_lon, 3))[:, :, 0])
ax[1].pcolormesh(lon[..., 0], lat[..., 0], B_pred.reshape((n_lat, n_lon, 3))[:, :, 1])
ax[2].pcolormesh(lon[..., 0], lat[..., 0], B_pred.reshape((n_lat, n_lon, 3))[:, :, 2])
# ax[1].pcolormesh(obs_lon, obs_lat, B_obs[:, 0].reshape(obs_lon.shape))
# ax[3].pcolormesh(obs_lon, obs_lat, B_obs[:, 1].reshape(obs_lon.shape))

ax[0].set_ylabel("Observations")
ax[0].set_title("B$_{X}$")
ax[2].set_title("B$_{Y}$")
plt.show()