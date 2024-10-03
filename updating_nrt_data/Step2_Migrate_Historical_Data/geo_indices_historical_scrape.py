import json
import logging
import urllib.request
from datetime import datetime, timedelta

import pandas as pd
import urllib3
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up InfluxDB
token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"
bucket = "indices_bucket"

client = InfluxDBClient(url=url, token=token, org=org, timeout=60_000, verify_ssl=False)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Get Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_data_from_api(start_time, end_time, index):
    url = f"https://kp.gfz-potsdam.de/app/json/?start={start_time}&end={end_time}&index={index}"
    log.info(f"Fetching data from API: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            log.info(f"Data fetched from API: {data}")
            return data
    except Exception as e:
        log.error(f"Error fetching data from API: {e}")
        return None


def process_data(data, index):
    times = data.get("datetime", [])
    values = data.get(index, [])
    if not times or not values:
        log.warning(f"No data found for index {index}")
        times = [datetime.utcnow()]
        values = [float("nan")]

    df = pd.DataFrame({"Time": pd.to_datetime(times), index: values})
    df.set_index("Time", inplace=True)
    return df


def save_to_influxdb(df, index, measurement_name, bucket_name):
    for time, row in df.iterrows():
        value = row[index] if pd.notnull(row[index]) else None
        point = (
            Point(measurement_name).time(time, WritePrecision.NS).field(index, value)
        )
        write_api.write(bucket=bucket_name, org=org, record=point)
        log.info(f"Saving point to InfluxDB: {point.to_line_protocol()}")
    log.info(f"Data saved to InfluxDB measurement '{measurement_name}'.")


def collect_and_save_data(start_date, end_date, index, interval_days=30):
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=interval_days), end_date)
        start_time_str = current_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = current_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        log.info(f"Start time: {start_time_str}, End time: {end_time_str}")

        data = get_data_from_api(start_time_str, end_time_str, index)
        if data:
            df = process_data(data, index)
            if df is not None:
                log.info(f"Processed DataFrame for {index}:")
                log.info(df.head())
                save_to_influxdb(df, index, "solar_indices", bucket)

        current_start = current_end


def main():
    start_date = datetime.strptime("2001-08-10", "%Y-%m-%d")
    end_date = datetime.strptime("2024-06-30", "%Y-%m-%d")
    # start_date = datetime.strptime('2024-07-01', '%Y-%m-%d')
    # end_date = datetime.strptime('2024-07-30', '%Y-%m-%d')
    indices = ["Kp", "Hp30", "ap30", "Fadj"]

    for index in indices:
        log.info(f"Collecting data for index {index}")
        collect_and_save_data(start_date, end_date, index)

    log.info("Data collection complete.")


if __name__ == "__main__":
    main()
