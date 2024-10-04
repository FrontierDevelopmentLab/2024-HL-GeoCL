import datetime as dt
import logging
import os
import pickle
import shutil
import warnings
from datetime import datetime

import dipole as dp
import numpy as np
import pandas as pd
import urllib3
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# Suppress specific FutureWarning from the ppigrf module
warnings.filterwarnings(
    action="ignore",
    message="The 'unit' keyword in TimedeltaIndex construction is deprecated",
    category=FutureWarning,
    module="ppigrf",  # Target the module where the warning originates
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up InfluxDB
token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"
# bucket = "ace_bucket"

client = InfluxDBClient(url=url, token=token, org=org, timeout=60_000, verify_ssl=False)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

# Get Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_earliest_data_point(influxdb_client, bucket, measurement):
    # Use the provided client to create a query client
    query_api = influxdb_client.query_api()

    # Define a more targeted query with an updated range if data is known to start in 2001
    query = f"""
    from(bucket: "{bucket}")
      |> range(start: 2000-01-01T00:00:00Z)  // Adjusted range start closer to known data start
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> first()
      |> keep(columns: ["_time"])
    """

    # Execute the query
    result = query_api.query(query=query)

    # Check if result is empty and handle appropriately
    if not result:
        print(f"No data found for {measurement} in {bucket}.")
        return None

    # Extract the first record's time if available
    for table in result:
        for record in table.records:
            return record.get_time()

    # If no records found, handle this scenario
    print(
        f"No records found for {measurement} in {bucket}. The data may not go back as far as queried."
    )
    return None


def format_datetime_for_filename(dt_str):
    # Parse the ISO 8601 datetime string to a datetime object
    date = dt.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    # Format the datetime object to a string suitable for filenames
    return date.strftime("%Y%m%dT%H%M%SZ")


def create_directory_structure():
    # Define the main directory
    main_dir = "saved_dataframes"

    # Define subdirectories
    sub_dirs = ["ace", "dscovr"]

    # Define sub-subdirectories for both 'ace' and 'dscovr'
    sub_sub_dirs_ace = [
        "feature_vectors_csv",
        "ace_indices_csv",
        "ace_indices_cleaned_csv",
        "ace_indices_pickle",
        "1_all",
    ]
    sub_sub_dirs_dscovr = [
        "feature_vectors_csv",
        "dscovr_indices_csv",
        "dscovr_indices_cleaned_csv",
        "dscovr_indices_pickle",
        "1_all",
    ]

    # Check and create the main directory
    os.makedirs(main_dir, exist_ok=True)

    # Create subdirectories within the main directory and their respective sub-subdirectories
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(main_dir, sub_dir)
        os.makedirs(sub_dir_path, exist_ok=True)  # Ensure subdirectory exists

        # Determine which set of sub-subdirectories to create based on the subdirectory name
        if sub_dir == "ace":
            sub_sub_dirs = sub_sub_dirs_ace
        else:
            sub_sub_dirs = sub_sub_dirs_dscovr

        # Create sub-subdirectories
        for sub_sub_dir in sub_sub_dirs:
            os.makedirs(os.path.join(sub_dir_path, sub_sub_dir), exist_ok=True)


def fetch_data(bucket, measurement, start, stop):
    query = f"""
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    tables = query_api.query(query=query)
    data = []
    for table in tables:
        for record in table.records:
            data.append(record.values)
    return pd.DataFrame(data)


def process_dataframe(df, output_path):
    if not df.empty:
        # Drop unnecessary columns if they exist
        columns_to_drop = ["result", "table", "_start", "_stop"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df.fillna(method="ffill", inplace=True, limit=10)

        if "_time" in df.columns:
            df.rename(columns={"_time": "Time"}, inplace=True)
            df.set_index("Time", inplace=True)
        else:
            print("The 'Time' column is not present in the DataFrame.")

        # Save to pickle
        df.to_pickle(output_path + ".pkl")

    else:
        print("DataFrame is empty.")


def load_and_display_pkl(file_path):
    """
    Load and display the contents of a pickle file.

    Parameters:
    file_path (str): The path to the pickle file.
    """
    try:
        # Open the file in binary read mode
        with open(file_path, "rb") as file:
            # Load the data from the file
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")


def calculate_and_merge_dipole(df):
    times = []
    tilts = []
    time = df.index[0]
    end_time = df.index[-1]

    # Calculate dipole tilt for each day within the data range
    while time <= end_time:
        fractional_year = (
            time.year + (time.month - 1) / 12
        )  # Correct calculation of fractional year
        tilt = dp.Dipole(fractional_year).tilt(
            time
        )  # Assuming correct usage of Dipole class
        times.append(time)
        tilts.append(tilt)
        time += dt.timedelta(days=1)

    # Create the DataFrame correctly
    dipole_tilt = pd.DataFrame({"tilt": tilts}, index=times)

    # Resample the tilt data to match the frequency of the input DataFrame
    dipole = dipole_tilt.resample("1T").ffill()

    # Merge the dipole data with the input DataFrame
    merged_data = df.join(dipole, how="left")
    merged_data["tilt"] = merged_data["tilt"].fillna(method="ffill")

    return merged_data


def process_instrument_feature_vector(df):
    feature_vectors = pd.DataFrame(index=df.index)
    feature_vectors["bx"] = df["bx_gsm"]
    feature_vectors["by"] = df["by_gsm"]
    feature_vectors["bz"] = df["bz_gsm"]
    feature_vectors["bt"] = df["bt"]
    feature_vectors["v"] = df["proton_speed"]
    feature_vectors["n"] = df["proton_density"]
    feature_vectors["t"] = df["proton_temperature"]

    feature_vectors["dipole_tilt"] = df["tilt"]

    feature_vectors["f107"] = df["Fadj"]
    feature_vectors["kp"] = df["Kp"]
    feature_vectors["hp30"] = df["Hp30"]
    feature_vectors["ap30"] = df["ap30"]

    feature_vectors["clock_angle"] = (
        np.arctan2(feature_vectors["by"], feature_vectors["bz"]) * 180 / np.pi
    )
    feature_vectors["sqrt_f107"] = np.sqrt(feature_vectors["f107"])
    feature_vectors["derived_1"] = feature_vectors["bt"] * np.cos(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_2"] = feature_vectors["v"] * np.cos(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_3"] = feature_vectors["dipole_tilt"] * np.cos(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_4"] = feature_vectors["sqrt_f107"] * np.cos(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_5"] = feature_vectors["bt"] * np.sin(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_6"] = feature_vectors["v"] * np.sin(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_7"] = feature_vectors["dipole_tilt"] * np.sin(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_8"] = feature_vectors["sqrt_f107"] * np.sin(
        feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_9"] = feature_vectors["bt"] * np.cos(
        2 * feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_10"] = feature_vectors["v"] * np.cos(
        2 * feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_11"] = feature_vectors["bt"] * np.sin(
        2 * feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_12"] = feature_vectors["v"] * np.sin(
        2 * feature_vectors["clock_angle"] * np.pi / 180
    )
    feature_vectors["derived_13"] = (
        np.sin(feature_vectors["clock_angle"] * np.pi / 180) ** 2
    )
    feature_vectors["p"] = (2 * 1e-6) * feature_vectors["n"] * feature_vectors["v"] ** 2
    feature_vectors["e"] = -feature_vectors["v"] * feature_vectors["bz"] * 1e-3
    return feature_vectors


def complete_data_process(influxdb_client, instrument, start, stop):
    # Define the buckets and measurements
    instrument_bucket = f"{instrument.lower()}_bucket"
    instrument_measurement = f"{instrument.lower()}_data"
    indices_bucket = "indices_bucket"
    indices_measurement = "solar_indices"

    # Fetch earliest available data point for both instrument and indices
    earliest_time = get_earliest_data_point(
        influxdb_client, instrument_bucket, instrument_measurement
    )
    earliest_time_indices = get_earliest_data_point(
        influxdb_client, indices_bucket, indices_measurement
    )

    # Convert start date to datetime object for comparison
    start_datetime = datetime.fromisoformat(start.replace("Z", "+00:00"))

    # Check availability of instrument and indices data
    if earliest_time is None or start_datetime < earliest_time:
        raise ValueError(
            f"OOPS! No data available for {instrument} starting from {start}. Earliest data starts at {earliest_time}."
        )
    if earliest_time_indices is None or start_datetime < earliest_time_indices:
        raise ValueError(
            f"OOPS! No data available for indices starting from {start}. Earliest data starts at {earliest_time_indices}."
        )

    formatted_start = format_datetime_for_filename(start)
    formatted_stop = format_datetime_for_filename(stop)

    # Fetch data for the instrument and indices
    instrument_data = fetch_data(instrument_bucket, instrument_measurement, start, stop)
    indices_data = fetch_data(
        indices_bucket, indices_measurement, start, stop
    )  # Static call

    # Process both datasets
    process_dataframe(
        instrument_data,
        f"saved_dataframes/{instrument.lower()}/1_all/{instrument.lower()}_data_filled_processed_{formatted_start}_{formatted_stop}",
    )
    process_dataframe(
        indices_data,
        f"saved_dataframes/{instrument.lower()}/1_all/indices_data_filled_processed_{formatted_start}_{formatted_stop}",
    )

    # Define source and destination file paths
    instrument_pickle = f"saved_dataframes/{instrument.lower()}/1_all/{instrument.lower()}_data_filled_processed_{formatted_start}_{formatted_stop}.pkl"
    instrument_destination_dir = (
        f"saved_dataframes/{instrument.lower()}/{instrument.lower()}_indices_pickle/"
    )

    indices_pickle = f"saved_dataframes/{instrument.lower()}/1_all/{instrument.lower()}_data_filled_processed_{formatted_start}_{formatted_stop}.pkl"

    instrument_destination_file = (
        f"{instrument.lower()}_{formatted_start}_{formatted_stop}.pkl"
    )
    indices_destination_file = f"indices_{formatted_start}_{formatted_stop}.pkl"
    # Ensure the destination directory exists
    os.makedirs(instrument_destination_dir, exist_ok=True)

    # Copy the file
    shutil.copy(
        os.path.join(instrument_pickle),
        os.path.join(instrument_destination_dir, instrument_destination_file),
    )
    shutil.copy(
        os.path.join(indices_pickle),
        os.path.join(instrument_destination_dir, indices_destination_file),
    )
    # Load and display processed data
    load_and_display_pkl(f"{instrument.lower()}_data_filled_processed.pkl")
    load_and_display_pkl("indices_data_filled_processed.pkl")

    # Load the processed data for further use
    instrument_nrt_data = pd.read_pickle(
        f"saved_dataframes/{instrument.lower()}/1_all/{instrument.lower()}_data_filled_processed_{formatted_start}_{formatted_stop}.pkl"
    )
    indices_nrt_data = pd.read_pickle(
        f"saved_dataframes/{instrument.lower()}/1_all/indices_data_filled_processed_{formatted_start}_{formatted_stop}.pkl"
    )

    # Prepare data
    indices_nrt_data["Fadj"] = indices_nrt_data["Fadj"].fillna(method="ffill")
    instrument_nrt_data.index = pd.to_datetime(instrument_nrt_data.index)
    indices_nrt_data.index = pd.to_datetime(indices_nrt_data.index)

    instrument_nrt_data.reset_index(inplace=True)
    indices_nrt_data.reset_index(inplace=True)

    instrument_nrt_data.sort_values("Time", inplace=True)
    indices_nrt_data.sort_values("Time", inplace=True)

    # Merge data using 'asof' merge
    merged_data = pd.merge_asof(
        instrument_nrt_data, indices_nrt_data, on="Time", direction="backward"
    )
    merged_data.set_index("Time", inplace=True)

    # Save the initial merged data
    merged_data.to_csv(
        f"saved_dataframes/{instrument.lower()}/1_all/{instrument.lower()}_indices_merged_data_{formatted_start}_{formatted_stop}.csv",
        index=True,
    )
    merged_data.to_csv(
        f"saved_dataframes/{instrument.lower()}/{instrument.lower()}_indices_csv/{formatted_start}_{formatted_stop}.csv",
        index=True,
    )

    # Clean up the merged data
    columns_to_remove = ["_measurement_x", "_measurement_y"]
    merged_data.drop(columns=columns_to_remove, inplace=True)
    merged_data.to_csv(
        f"saved_dataframes/{instrument.lower()}/1_all/cleaned_{instrument.lower()}_indices_merged_data_{formatted_start}_{formatted_stop}.csv",
        index=True,
    )
    merged_data.to_csv(
        f"saved_dataframes/{instrument.lower()}/{instrument.lower()}_indices_cleaned_csv/{formatted_start}_{formatted_stop}.csv",
        index=True,
    )

    # Calculate dipole tilt and merge it with the data
    final_merged_data = calculate_and_merge_dipole(merged_data)

    # Process the final instrument data and save to CSV
    final_feature_vector = process_instrument_feature_vector(
        final_merged_data
    )  # This function needs to exist
    final_feature_vector.to_csv(
        f"saved_dataframes/{instrument.lower()}/1_all/{instrument}_indices_feature_vectors_{formatted_start}_{formatted_stop}.csv",
        index=True,
    )
    final_feature_vector.to_csv(
        f"saved_dataframes/{instrument.lower()}/feature_vectors_csv/{formatted_start}_{formatted_stop}.csv",
        index=True,
    )


"""''
earliest_time_ace = get_earliest_data_point(client, 'ace_bucket', 'ace_data')
earliest_time_dscovr = get_earliest_data_point(client, 'dscovr_bucket', 'dscovr_data')
earliest_indices_time = get_earliest_data_point(client, 'indices_bucket', 'solar_indices')

print("Earliest ACE record timestamp:", earliest_time_ace)
print("Earliest DSCOVR record timestamp:", earliest_time_dscovr)
print("Earliest Geomagnetic Indices record timestamp:", earliest_indices_time)
"""

# create_directory_structure()

try:
    complete_data_process(
        client, "DSCOVR", "2001-08-07T00:00:00Z", "2001-08-07T23:59:59Z"
    )
except ValueError as e:
    print(e)

# complete_data_process('ACE', '2016-07-26T00:00:00Z', '2016-07-26T23:59:59Z')
# complete_data_process('DSCOVR', '2016-07-26T00:00:00Z', '2016-07-26T23:59:59Z')
