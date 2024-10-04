"""A module that plots the SECs + GP mean and standard deviation for SuperMAG stations
            ==> will be later deployed on our website

Author: Opal Issan (oissan@ucsd.edu) PhD @ ucsd
        India Jackson (indiajacksonphd@gmail.com) Posdoc @ gsu


Last Modified: July 30th, 2024
"""

import cartopy.crs as ccrs
import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import stripy

# import supermag_api as smapi
from sec import T_df, get_mesh, remove_duplicate_lonlat

matplotlib.use("TkAgg")

font = {"family": "serif", "size": 14}

matplotlib.rc("font", **font)
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

# # read in SuperMAG data using api (we will have a more efficient approach for website near-real-time)
# # see supermag_api.py for more details
# # start date year-month-day-hour-min-sec
# start = [2018, 3, 17, 18, 40, 0]
# # read in SuperMAG data
# (status, stations) = smapi.SuperMAGGetInventory("opaliss", start, 3600)
# # number of stations available at this time
# n_stations = len(stations)
# print("number of stations =", n_stations)
# # initialize magnetic data and SuperMAG station locations
# data_Bn, data_Be, geo_lat, geo_lon = np.zeros(n_stations), np.zeros(n_stations), np.zeros(n_stations), np.zeros(n_stations)
#
#
# # note: this is very slow... we will use something better thanks to the one and only @bibhuti
# kk = 0
# for ii in range(0, n_stations):
#     (status, sm_data) = amapi.SuperMAGGetData("opaliss", start, 3600, 'geo', stations[ii])
#     try:
#         print("kk = ", kk)
#         if sm_data.glat[0] not in geo_lat and sm_data.glon[0] not in geo_lon:
#             data_Bn[kk] = sm_data.N[0]["geo"]
#             data_Be[kk] = sm_data.E[0]["geo"]
#             geo_lat[kk] = sm_data.glat[0]
#             geo_lon[kk] = sm_data.glon[0]
#             kk += 1
#     except:
#         print("An exception occurred at " + str(stations[ii]))
#
# data_Bn, data_Be, data_Bz, geo_lat, geo_lon = data_Bn[:kk], data_Be[:kk], data_Bz[:kk], geo_lat[:kk], geo_lon[:kk]
# # save SuperMAG data
# np.save("data/Bn.npy", data_Bn)
# np.save("data/Be.npy", data_Be)
# np.save("data/geo_lat.npy", geo_lat)
# np.save("data/geo_lon.npy", geo_lon)
# load SuperMAG data
data_Bn = np.load("data/Bn.npy")
data_Be = np.load("data/Be.npy")
geo_lat = np.load("data/geo_lat.npy")
geo_lon = np.load("data/geo_lon.npy")

# set up constants for SECs
R_earth = 6371  # in km
R_ionosphere = R_earth + 100  # in km

# setup the SECs "node" grid
# n_lon and n_lat are free parameters but are limited to n_lon*n_lat ~ number of stations
lon, lat = remove_duplicate_lonlat(lon=geo_lon, lat=geo_lat)
mesh = stripy.sTriangulation(
    lon * np.pi / 180 - np.pi * 0.99, lat * np.pi / 180, refinement_levels=1
)
mesh_lat = mesh.lats[::4] * 180 / np.pi
mesh_lon = mesh.lons[::4] * 180 / np.pi + 180
# specify the secs grid
secs_lat_lon_r = np.hstack(
    (
        mesh_lat.reshape(-1, 1),
        mesh_lon.reshape(-1, 1),
        R_ionosphere * np.ones(len(mesh_lat)).reshape(-1, 1),
    )
)
# setup the SuperMAG stations grid
obs_lat_lon_r = np.vstack((geo_lat, geo_lon, R_earth * np.ones(len(geo_lon)))).T

# observations in a vector
B_obs = np.append(data_Bn, data_Be)
B_obs = np.reshape(B_obs, (len(B_obs), 1))

# get T matrix for SECs
T_mat = T_df(obs_loc=obs_lat_lon_r, sec_loc=secs_lat_lon_r, include_Bz=False)

# setup GP kernel + its hyperparameters
kernel = GPy.kern.Linear(input_dim=np.shape(T_mat)[1], variances=1)

# create simple GP model
model = GPy.models.GPRegression(T_mat, B_obs, kernel)
model.Gaussian_noise.constrain_bounded(0, 200)
# optimize GP hyperparameters
model.optimize(messages=True)

# predicted grid
n_lat, n_lon = 100, 200
pred_lat_lon_r, pred_lat, pred_lon = get_mesh(
    n_lon=n_lon, n_lat=n_lat, radius=R_earth, lat_max=80, lat_min=-80, endpoint_lon=True
)
# predict via GP
mean_, sd_ = model.predict(
    Xnew=T_df(obs_loc=pred_lat_lon_r, sec_loc=secs_lat_lon_r, include_Bz=False)
)

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(
    pred_lon[:, :, 0],
    pred_lat[:, :, 0],
    np.reshape(mean_[: n_lat * n_lon], (n_lat, n_lon), "C"),
    alpha=0.6,
    transform=ccrs.PlateCarree(),
)

ax.scatter(
    geo_lon,
    geo_lat,
    c=data_Bn,
    vmin=np.min(mean_[: n_lat * n_lon]),
    vmax=np.max(mean_[: n_lat * n_lon]),
    s=10,
    cmap="viridis",
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("mean $B_{n}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig("figures/secgp_mean_Bn.png", bbox_inches="tight", dpi=600)
plt.show()

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(
    pred_lon[:, :, 0],
    pred_lat[:, :, 0],
    np.reshape(mean_[n_lat * n_lon : 2 * n_lat * n_lon], (n_lat, n_lon), "C"),
    alpha=0.6,
    transform=ccrs.PlateCarree(),
)
ax.scatter(
    geo_lon,
    geo_lat,
    c=data_Be,
    vmin=np.min(mean_[n_lat * n_lon : 2 * n_lat * n_lon]),
    vmax=np.max(mean_[n_lat * n_lon : 2 * n_lat * n_lon]),
    s=10,
    cmap="viridis",
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("mean $B_{e}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig("figures/secgp_mean_Be.png", bbox_inches="tight", dpi=600)
plt.show()

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(
    pred_lon[:, :, 0],
    pred_lat[:, :, 0],
    np.reshape(np.sqrt(sd_)[: n_lat * n_lon], (n_lat, n_lon), "C"),
    alpha=0.6,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)

ax.scatter(mesh_lon, mesh_lat, c="blue", s=7, marker="x")
ax.scatter(geo_lon, geo_lat, c="red", s=7)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("standard deviation $B_{n}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig("figures/secgp_sd_Bn.png", bbox_inches="tight", dpi=600)
plt.show()

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(
    pred_lon[:, :, 0],
    pred_lat[:, :, 0],
    np.reshape(np.sqrt(sd_)[n_lat * n_lon : 2 * n_lat * n_lon], (n_lat, n_lon), "C"),
    alpha=0.6,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)

ax.scatter(mesh_lon, mesh_lat, c="blue", s=7, marker="x")
ax.scatter(geo_lon, geo_lat, c="red", s=7)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("standard deviation $B_{e}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig("figures/secgp_sd_Be.png", bbox_inches="tight", dpi=600)
plt.show()
