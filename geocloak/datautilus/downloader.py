"""
This module provides an API to download various near real-time data
from multiple observatories and sources using their endpoint APIs.
It also allows the user to choose to save the data in a specific format.
"""

import datetime
import gzip
import os
import urllib
from io import BytesIO

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

from geocloak.configs.nrtdata import dscovr_f1m_cols, dscovr_m1m_cols


class Downloader:
    """
    This python class will download the various data set for a given url.
    """

    def __init__(self, observatory="DSCVR"):
        self.observatory = observatory

    @staticmethod
    def _download_dscvr(file_url: str, **kwargs) -> pd.DataFrame:
        """
        A Helper function to download the data from DSCVR website and convert it to dataframe.

        Parameters
        ----------
        file_url: str
            The url of the downlaodable file.

        Returns
        -------
        df: pd.DataFrame
            The dataframe containing the data.

        """
        data_list = []
        data_type = kwargs.get("data_type")
        cols = []

        # Look for the nature of data
        if data_type == "m1m":
            cols = dscovr_m1m_cols
        elif data_type == "f1m":
            cols = dscovr_f1m_cols

        # Downlaod and read data and convert it to dataframe
        with urllib.request.urlopen(file_url, timeout=40) as resp:
            with gzip.open(BytesIO(resp.read())) as gz:
                with nc.Dataset("inmemory.nc", memory=gz.read()) as f:
                    idx = nc.num2date(f.variables["time"][:], f.variables["time"].units)
                    for col in cols:
                        vals = np.asarray(f.variables[col][:])
                        data_list.append(pd.Series(vals, index=idx))
        df = pd.DataFrame(data_list).transpose()
        df.columns = cols
        return df

    def download(self, file_url: str, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        file_url

        Returns
        -------

        """
        df = None
        if self.observatory == "DSCVR":
            df = self._download_dscvr(file_url, **kwargs)
        else:
            NotImplemented("Not implemented yet.")
        return df


class NRTDataDownloader:
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


    methods
    --------
    get_ACEdata(baseurl: str = None, outformat=None) -> None
        To get ACE data.
    get_DSCOVRdata(baseurl: str = None, outformat=None) -> None
        To get DSCOVR data.

    """

    def __init__(self, path="data", download: bool = False):
        """Class initiation"""
        self.path = path
        self.download = download
        os.makedirs(path, exist_ok=True)
        self._dscovr_url = (
            "https://www.ngdc.noaa.gov/next-catalogs/rest/dscovr/catalog/"
        )
        self._omni_url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

    def __repr__(self) -> str:
        return f"NRTDataDownloader(path={self.path}, download={self.download})"

    def __str__(self) -> str:
        return f"NRTDataDownloader(path={self.path}, download={self.download})"

    def check_data_status(self) -> tuple:
        """
        To check the current status of NRT data.

        Return
        -------
        (Lastdata, latestTime) : tuple
            Time of the last data download and latest data available.
        """
        # get time information about the data
        time_now = datetime.datetime.now().strftime("%Y-%m-%dT00:00Z")
        time_lastdata = datetime.datetime(2015, 1, 1).strftime("%Y-%m-%dT00:00Z")
        query_url = "extents?processEnvs=oe&dataTypes=f1m,m1m&dataStartTime={}&dataEndTime={}".format(
            time_lastdata, time_now
        )

        # get query url
        url = self._dscovr_url + query_url
        response = requests.get(url)
        if response.ok:
            response = response.json()
            latestTime = response["latestTime"]
            return time_lastdata, latestTime
        else:
            return None, None

    def get_ACEdata(self, baseurl: str = None, outformat=None) -> None:
        """
        This method will check for the existing ACE data and download rest of NRT data.

        Parameters
        ----------
        baseurl : str, optional
            This is the base url for the ACE data, by default None
        outformat : str, optional
            The outformat of the data, by default None

        Return
        -------
            None
        """

        pass

    def get_DSCOVRdata(
            self,
            start_date: datetime.datetime = None,
            end_date: datetime.datetime = None,
            outformat=None,
    ) -> pd.DataFrame | None:
        """
        This method will check for exiting DSCOVR data and download rest of NRT data.

        Parameters
        ----------
        start_date: datetime.datetime, optional
            Start date of the data range, by default None.
        end_date: datetime.datetime, optional
            End date of the data range, by default None.
        outformat : str, optional
            The outformat of the data, by default None

        Return
        -------
            None
        """
        downloader = Downloader()
        base_url = "https://www.ngdc.noaa.gov/dscovr/data/{}/{}/{}"

        # get current status of the data
        last_data, latestTime = self.check_data_status()
        if start_date is None:
            start_date = last_data
        if end_date is None:
            end_date = latestTime
        query_url = "filename?processEnvs=oe&dataTypes=f1m,m1m&dataStartTime={}&dataEndTime={}".format(
            start_date, end_date
        )

        # Create list of urls for DSCOVR data set
        url = self._dscovr_url + query_url
        response = requests.get(url)
        filenames = None
        if response.ok:
            response = response.json()
            filenames = response["items"]
        if filenames is None:
            return None
        dflistf1m = []
        dflistm1m = []

        # Get all the data in list of data frames
        for filename in filenames:
            ind = filename.find("_s")
            yr = filename[ind + 2: ind + 6]
            mo = filename[ind + 6: ind + 8]
            downlaod_url = base_url.format(yr, mo, filename)
            if filename.startswith("oe_f1m_dscovr_"):
                dflistf1m.append(downloader.download(downlaod_url, data_type="f1m"))
            elif filename.startswith("oe_m1m_dscovr_"):
                dflistm1m.append(downloader.download(downlaod_url, data_type="m1m"))

        # Merge list of data frames
        merged_df_f1m = pd.concat(dflistf1m)
        merged_df_m1m = pd.concat(dflistm1m)
        filtered_df_f1m = merged_df_f1m[~merged_df_f1m.index.duplicated(keep="first")]
        filtered_df_m1m = merged_df_m1m[~merged_df_m1m.index.duplicated(keep="first")]
        filtered_df_f1m.index.name = "Time"
        filtered_df_m1m.index.name = "Time"
        ind = filtered_df_f1m.index.intersection(filtered_df_m1m.index)

        # Make sure that size of two data frames are same
        assert filtered_df_f1m.shape[0] == filtered_df_m1m.shape[0]
        df = filtered_df_f1m.join(filtered_df_m1m, how="inner")

        return df

    def get_SuperMAGdata(self, baseurl: str = None, outformat=None) -> None:
        """
        This method will check for exiting SuperMAG data and download rest of NRT data.

        Parameters
        ----------
        baseurl : str, optional
            This is the base url for the ACE data, by default None
        outformat : str, optional
            The outformat of the data, by default None

        Return
        -------
            None
        """
