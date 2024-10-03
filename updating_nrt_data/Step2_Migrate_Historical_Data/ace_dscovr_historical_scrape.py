import logging
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import urllib3
import requests
import h5py
from io import BytesIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up InfluxDB
token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

client = InfluxDBClient(url=url, token=token, org=org, timeout=60_000, verify_ssl=False)
write_api = client.write_api(write_options=SYNCHRONOUS)

def download_h5_from_gcs(url):
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

def save_to_influxdb(df, measurement_name, bucket_name):
    for time, row in df.iterrows():
        point = Point(measurement_name) \
            .time(time, WritePrecision.NS) \
            .field("bt", row['Bt']) \
            .field("bx_gsm", row['Bx']) \
            .field("by_gsm", row['By']) \
            .field("bz_gsm", row['Bz']) \
            .field("proton_speed", row['Speed']) \
            .field("proton_density", row['Density']) \
            .field("proton_temperature", row['Temperature'])
        write_api.write(bucket=bucket_name, org=org, record=point)

def collect_and_save_data(year, dataset):
    base_url = 'https://storage.cloud.google.com/geocloak2024/formatted_data/{dataset}/{dataset}_1m/{dataset}_formatted_1m_{year}.h5'
    url = base_url.format(dataset=dataset, year=year)
    h5_data = download_h5_from_gcs(url)

    # Open the HDF5 file
    with h5py.File(h5_data, 'r') as hdf:
        # Extracting data
        labels = [label.decode('utf-8') for label in hdf['data/axis0'][:]]  # Decode byte strings
        timestamps = hdf['data/axis1'][:]
        data_values = hdf['data/block0_values'][:]

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(timestamps)

        # Create a DataFrame
        df = pd.DataFrame(data=data_values, index=timestamps, columns=labels)
        df_reset = df.reset_index().rename(columns={'index': 'Timestamp'})

    # Save to InfluxDB
    save_to_influxdb(df, f"{dataset}_data", f"{dataset}_bucket")
    logging.info(f"Data for year {year} saved to InfluxDB under {dataset}_data")

# Example usage
collect_and_save_data(2016, "ACE")
