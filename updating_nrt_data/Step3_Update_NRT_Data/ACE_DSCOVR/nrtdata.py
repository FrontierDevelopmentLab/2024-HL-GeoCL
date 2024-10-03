import logging

import pandas as pd
import urllib3
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get Logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# InfluxDB Setup
token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

client = InfluxDBClient(url=url, token=token, org=org, timeout=60_000, verify_ssl=False)
write_api = client.write_api(write_options=SYNCHRONOUS)


class NRTData:
    """Class to download NRT data from various sources."""

    rtsw_files = ["rtsw_mag_1m.json", "rtsw_wind_1m.json"]

    def __init__(self):
        """Class initiation"""
        self.rtsw_url = "https://services.swpc.noaa.gov/json/rtsw/{}"
        self.data = None

    def __rtsw(self):
        """Internal helper function to get the Real Time Data."""
        try:
            dfmag = pd.read_json(self.rtsw_url.format(self.rtsw_files[0]))
            dfmag.set_index("time_tag", inplace=True)
            dfmag = dfmag[["bt", "bx_gsm", "by_gsm", "bz_gsm", "source"]]
            dfsw = pd.read_json(self.rtsw_url.format(self.rtsw_files[1]))
            dfsw.set_index("time_tag", inplace=True)
            dfsw = dfsw[
                ["proton_speed", "proton_density", "proton_temperature", "source"]
            ]
        except FileNotFoundError:
            log.warning("The Real Time file is not available.")
            return None
        return [dfmag, dfsw]

    @property
    def nrtACE(self):
        """To get Real Time Solar Wind Data from ACE."""
        data = self.__rtsw()
        if data is None:
            log.warning("No Real Time Data is available.")
            return None
        _df1 = data[0][data[0].source == "ACE"].copy()
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = data[1][data[1].source == "ACE"].copy()
        _df2.drop(["source"], axis=1, inplace=True)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"
        return _df

    @property
    def nrtDSCOVER(self):
        """To get Real Time Solar Wind Data from DSCOVR."""
        data = self.__rtsw()
        if data is None:
            log.warning("No Real Time Data is available.")
            return None

        _df1 = data[0][data[0].source == "DSCOVR"].copy()
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = data[1][data[1].source == "DSCOVR"].copy()
        _df2.drop(["source"], axis=1, inplace=True)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"
        return _df

    def save_to_influxdb(self, df, measurement_name, bucket_name):
        for time, row in df.iterrows():
            point = (
                Point(measurement_name)
                .time(time, WritePrecision.NS)
                .field("bt", row["bt"])
                .field("bx_gsm", row["bx_gsm"])
                .field("by_gsm", row["by_gsm"])
                .field("bz_gsm", row["bz_gsm"])
                .field("proton_speed", row["proton_speed"])
                .field("proton_density", row["proton_density"])
                .field("proton_temperature", row["proton_temperature"])
            )
            write_api.write(bucket=bucket_name, org=org, record=point)
        log.info(f"Data saved to InfluxDB measurement '{measurement_name}'.")
