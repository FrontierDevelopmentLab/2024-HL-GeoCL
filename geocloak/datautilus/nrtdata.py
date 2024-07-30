"""
Python Module to download the NRT data from NOAA website.
"""

import logging
import os
import pathlib

import pandas as pd

from geocloak.configs.datainfo import dscovr_f1m_cols, dscovr_m1m_cols

this_dir = pathlib.Path(__file__).parent.absolute()

# Get Logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class NRTData:
    """Class to download NRT data form various sources.

    Parameters
    --------
    path: str, optional
        Output path of the downloaded data. Defaults to "data".
    download: bool, optional
        Whether to download the data or not. Defaults to False.

    Attributes
    ---------
    path: str
        Output path of the downloaded data.
    download: bool
        Whether to download the data or not.
    """

    last_data_time = {"dscovr": None}
    rtsw_files = ["rtsw_mag_1m.json", "rtsw_wind_1m.json"]

    def __init__(self, path="data", download: bool = False):
        """Class initiation"""
        self.path = path
        self.download = download
        os.makedirs(path, exist_ok=True)
        self.rtsw_url = "https://services.swpc.noaa.gov/json/rtsw/{}"
        self.data = None

    def __repr__(self) -> str:
        return f"NRTData(path={self.path}, download={self.download})"

    def __str__(self) -> str:
        return f"NRTData(path={self.path}, download={self.download})"

    def __rtsw(self):
        """Internal helper function to get the Real Time Data.

        Returns
        -------
        pd.DataFrame | None:
            The Real Time Data in pandas dataframe.

        """
        try:
            dfmag = pd.read_json(self.rtsw_url.format(self.rtsw_files[0]))
            dfmag.set_index("time_tag", inplace=True)
            dfmag = dfmag[dscovr_m1m_cols + ["source"]]
            dfsw = pd.read_json(self.rtsw_url.format(self.rtsw_files[1]))
            dfsw.set_index("time_tag", inplace=True)
            dfsw = dfsw[dscovr_f1m_cols + ["source"]]
        except FileNotFoundError:
            log.warning("The Real Time file is not available.")
            return None
        data = [dfmag, dfsw]
        return data

    @property
    def nrtACE(self):
        """To get Real Time Solar Wind Data.

        Returns
        --------
        _df: pd.DataFrame | None
            Dataframe containing the NRT ACE data.
        """
        data = self.__rtsw()
        # if no data found
        if data is None:
            log.warning("No Real Time Data is available.")
            return None
        _df1 = data[0][data[0].source == "ACE"]
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = data[1][data[1].source == "ACE"]
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

        _df1 = data[0][data[0].source == "DSCOVR"]
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = data[1][data[1].source == "DSCOVR"]
        _df2.drop(["source"], axis=1, inplace=True)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"
        return _df

    @property
    def nrtboth(self):
        """Property to get combined data from both the observatories.

        Returns
        -------
        _df: pd.DataFrame | None
            Dataframe containing the NRT ACE and DSCOVR data.
        """
        data = self.__rtsw()
        if data is None:
            log.warning("No Real Time Data is available.")
            return None
        _df1 = data[0]
        _df2 = data[1].drop(["source"], axis=1)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"
        return _df
