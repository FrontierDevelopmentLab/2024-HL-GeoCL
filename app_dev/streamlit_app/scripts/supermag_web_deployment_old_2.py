"""A module that plots the SECs + GP mean and standard deviation for SuperMAG stations
            ==> will be later deployed on our website

Author: Opal Issan (oissan@ucsd.edu) PhD @ ucsd
        India Jackson (indiajacksonphd@gmail.com) Posdoc @ gsu


Last Modified: July 30th, 2024
"""

import os
from io import BytesIO

import cartopy.crs as ccrs
import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sec import T_df, get_mesh


def supermag_setup():
    os.environ.setdefault("GCLOUD_PROJECT", "hl-geo'")

    # Set your project ID
    project_id = "hl-geo"

    # Initialize a client
    storage_client = storage.Client(project=project_id)

    font = {"family": "serif", "size": 14}

    matplotlib.rc("font", **font)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)

    storage_client = storage.Client()
    return storage_client


def download_csv_from_gcs(storage_client, bucket_name, source_blob_name):
    """Downloads a blob from the bucket and returns it as a pandas DataFrame."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    content = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(content), header=None)
    return df


'''
# Function to convert CSV to NPY
def csv_to_npy(csv_file_path):
    """Convert a .csv file to a numpy array."""
    df = pd.read_csv(csv_file_path)
    data = df.to_numpy()
    return data
'''


# Function to convert DataFrame to NPY
def df_to_npy(df):
    """Convert a pandas DataFrame to a numpy array."""
    return df.to_numpy()


'''
# Function to split lat/lon CSV into separate NPY files
def split_lat_lon_csv_to_npy(csv_file_path, lat_npy_file, lon_npy_file):
    """Split lat/lon data from CSV and save as separate NPY files."""
    df = pd.read_csv(csv_file_path)
    glat = df['glat'].to_numpy()
    glon = df['glon'].to_numpy()
'''


# Function to split lat/lon DataFrame into separate NPY arrays
def split_lat_lon_df(df):
    """Split lat/lon DataFrame and return separate NPY arrays."""
    glat = df["glat"].to_numpy()
    glon = df["glon"].to_numpy()
    return glat, glon


def prepare_supermag(storage_client):
    bucket_name = "geocloak2024"
    csv_files = {
        "inference_outputs/DAGGER/202405120000/dbn_geo/202405120000.csv": "data_Bn.npy",
        "inference_outputs/DAGGER/202405120000/dbe_geo/202405120000.csv": "data_Be.npy",
        "inference_outputs/DAGGER/202405120000/dbz_geo/202405120000.csv": "data_Bz.npy",
    }

    lat_lon_csv = "formatted_data/SuperMAG/supermag_processed/stns.csv"

    npy_data = {}

    # Download and convert each CSV file
    for csv_file, npy_var in csv_files.items():
        df = download_csv_from_gcs(storage_client, bucket_name, csv_file)
        npy_data[npy_var] = df_to_npy(df)

    # Assign the NumPy arrays to variables
    data_Bn = npy_data["data_Bn.npy"]
    data_Be = npy_data["data_Be.npy"]
    data_Bz = npy_data["data_Bz.npy"]

    # Download and split lat/lon CSV

    lat_lon_df = download_csv_from_gcs(storage_client, bucket_name, lat_lon_csv)
    geo_lat, geo_lon = split_lat_lon_df(lat_lon_df)

    """
    # Load the lat/lon data
    geo_lat = np.load("data/glat.npy")
    geo_lon = np.load("data/glon.npy")

    geo_lat = npy_data["geo_lat.npy"]
    geo_lon = npy_data["geo_lon.npy"]

    # Optional: Save the NumPy arrays locally
    for npy_file, data in npy_data.items():
        np.save(f"data/{npy_file}", data)
        print(f"Saved {npy_file} to data directory.")
    """

    # Print shapes to verify the data
    print(f"data_Bn shape: {data_Bn.shape}")
    print(f"data_Be shape: {data_Be.shape}")
    print(f"data_Bz shape: {data_Bz.shape}")
    print(f"geo_lat shape: {geo_lat.shape}")
    print(f"geo_lon shape: {geo_lon.shape}")

    ############################################################################################
    # Setup the SuperMAG stations grid
    min_length = min(
        len(data_Bn), len(data_Be), len(data_Bz), len(geo_lat), len(geo_lon)
    )

    data_Bn = data_Bn[:min_length]
    data_Be = data_Be[:min_length]
    data_Bz = data_Bz[:min_length]
    geo_lat = geo_lat[:min_length]
    geo_lon = geo_lon[:min_length]
    ############################################################################################
    return data_Bn, data_Be, data_Bz, geo_lat, geo_lon


def process_supermag_data(
    data_Bn, data_Be, data_Bz, geo_lat, geo_lon, secs_lat_lon_r, R_earth
):

    # setup the SuperMAG stations grid
    obs_lat_lon_r = np.vstack((geo_lat, geo_lon, R_earth * np.ones(len(geo_lon)))).T

    # observations in a vector
    B_obs = np.append(np.append(data_Bn, data_Be), data_Bz)
    B_obs = np.reshape(B_obs, (len(B_obs), 1))

    # get T matrix for SECs
    T_mat = T_df(obs_loc=obs_lat_lon_r, sec_loc=secs_lat_lon_r)

    print(f"Shape of X: {T_mat.shape}")
    print(f"Shape of Y: {B_obs.shape}")

    # setup GP kernel + its hyperparameters
    kernel = GPy.kern.Linear(input_dim=np.shape(T_mat)[1], variances=1)

    # create simple GP model
    model = GPy.models.GPRegression(T_mat, B_obs, kernel)

    # optimize GP hyperparameters
    model.optimize(messages=True)

    # predicted grid
    n_lat, n_lon = 100, 200
    pred_lat_lon_r, pred_lat, pred_lon = get_mesh(
        n_lon=n_lon,
        n_lat=n_lat,
        radius=R_earth,
        lat_max=80,
        lat_min=-80,
        endpoint_lon=True,
    )
    # predict via GP
    mean_, sd_ = model.predict(
        Xnew=T_df(obs_loc=pred_lat_lon_r, sec_loc=secs_lat_lon_r)
    )
    return mean_, sd_, pred_lat, pred_lon, n_lat, n_lon


def plot_supermag_results(
    mean_,
    sd_,
    pred_lat,
    pred_lon,
    n_lat,
    n_lon,
    geo_lat,
    geo_lon,
    data_Bn,
    data_Be,
    lon_sec,
    lat_sec,
):
    figs = []

    # Plot mean Bn
    fig, ax = plt.subplots(
        figsize=(9, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
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
    figs.append(fig)

    # Plot mean Be
    fig, ax = plt.subplots(
        figsize=(9, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
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
    figs.append(fig)

    # Plot standard deviation Bn
    fig, ax = plt.subplots(
        figsize=(9, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    pos = ax.contourf(
        pred_lon[:, :, 0],
        pred_lat[:, :, 0],
        np.reshape(np.sqrt(sd_)[: n_lat * n_lon], (n_lat, n_lon), "C"),
        alpha=0.6,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(lon_sec[:, :, 0], lat_sec[:, :, 0], c="blue", s=7, marker="x")
    ax.scatter(geo_lon, geo_lat, c="red", s=7)
    cbar = fig.colorbar(pos)
    cbar.ax.set_ylabel("standard deviation $B_{n}$ [nT]", rotation=90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-80, -40, 0, 40, 80])
    ax.set_ylim(-80, 80)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    figs.append(fig)

    # Plot standard deviation Be
    fig, ax = plt.subplots(
        figsize=(9, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    pos = ax.contourf(
        pred_lon[:, :, 0],
        pred_lat[:, :, 0],
        np.reshape(
            np.sqrt(sd_)[n_lat * n_lon : 2 * n_lat * n_lon], (n_lat, n_lon), "C"
        ),
        alpha=0.6,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(lon_sec[:, :, 0], lat_sec[:, :, 0], c="blue", s=7, marker="x")
    ax.scatter(geo_lon, geo_lat, c="red", s=7)
    cbar = fig.colorbar(pos)
    cbar.ax.set_ylabel("standard deviation $B_{e}$ [nT]", rotation=90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-80, -40, 0, 40, 80])
    ax.set_ylim(-80, 80)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    figs.append(fig)

    return figs


# set up constants for SECs
R_earth = 6371  # in km
R_ionosphere = R_earth + 100  # in km

# setup the SECs "node" grid
# n_lon and n_lat are free parameters but are limited to n_lon*n_lat ~ number of stations
n_lon, n_lat = 35, 35
secs_lat_lon_r, lat_sec, lon_sec = get_mesh(
    n_lon=n_lon, n_lat=n_lat, radius=R_ionosphere
)

client = supermag_setup()
data_Bn, data_Be, data_Bz, geo_lat, geo_lon = prepare_supermag(client)
# Process the data
mean_, sd_, pred_lat, pred_lon, n_lat, n_lon = process_supermag_data(
    data_Bn, data_Be, data_Bz, geo_lat, geo_lon, secs_lat_lon_r, R_earth
)

# Plot the results and display in Streamlit
figs = plot_supermag_results(
    mean_,
    sd_,
    pred_lat,
    pred_lon,
    n_lat,
    n_lon,
    geo_lat,
    geo_lon,
    data_Bn,
    data_Be,
    lon_sec,
    lat_sec,
)

for fig in figs:
    plt.show()
