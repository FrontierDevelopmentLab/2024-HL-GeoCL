"""
This module provides an API to download various exiting data from the
from multiple observatories and sources using their endpoint APIs.
"""

import logging
import datetime
import gzip
import os
import urllib
from io import BytesIO

import netCDF4 as nc
import numpy as np
import pandas as pd
import tqdm
import requests

from geocloak.configs.datainfo import dscovr_f1m_cols, dscovr_m1m_cols


def _download_dscvr(file_url: str, **kwargs) -> pd.DataFrame | None:
    """
    A Helper function to download the data from DSCVR website and convert it to dataframe.

    Parameters
    ----------
    file_url: str
        The url of the downlaodable file.
    **kwargs: dict
        Keywords about the data series based on which the
        list of colums from the NetCDF file will be selcted.

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the data.

    """
    data_list = []
    data_type = kwargs.get("data_type")
    logger = kwargs.get("logger")
    cols = []

    # Look for the nature of data
    if data_type == "m1m":
        cols = dscovr_m1m_cols
    elif data_type == "f1m":
        cols = dscovr_f1m_cols

    # Downlaod and read data and convert it to dataframe
    try:
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
    except Exception as e:
        if logger is not None:
            logger.error(f"Unable to download the data from DSCVR website.")
        return None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data Downloader Class
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class DataDownloader:
    """Downlaod all the exiting data from various sources and save them
    in pandas HDF format for AI/ML applications.

    Attributes
    -----------
    outpath: str
        The path to the directory where the data will be saved.

    Methods
    -----------
    get_data

    """

    def __init__(self, outpath: str = None):
        self.outpath = outpath
        if self.outpath is None:
            self.outpath = "data"
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath, exist_ok=True)

        # Log handler
        # Create a custom logger
        self.logger = logging.getLogger(__name__)

        # Create handlers
        f_handler = logging.FileHandler("log.log")
        f_handler.setLevel(logging.WARNING)

        # Create formatters and add it to handlers
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(f_handler)

    def _dscovr(
        self,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        url_only=False,
    ) -> pd.DataFrame | None:
        """
        This method will check for exiting DSCOVR data and download rest of NRT data.

        Parameters
        ----------
        start_date: datetime.datetime, optional
            Start date of the data range, by default None.
        end_date: datetime.datetime, optional
            End date of the data range, by default None.
        url_only: bool, optional
            Whether or not to download the data from URL, by default False.

        Return
        -------
            None
        """
        base_url = "https://www.ngdc.noaa.gov/dscovr/data/{}/{}/{}"
        query_url = "https://www.ngdc.noaa.gov/next-catalogs/rest/dscovr/catalog/"

        # get current status of the data
        if start_date is None:
            start_date = datetime.datetime(2015, 1, 1).strftime("%Y-%m-%dT00:00Z")
        else:
            start_date = start_date.strftime("%Y-%m-%dT00:00Z")
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%dT00:00Z")
        else:
            end_date = end_date.strftime("%Y-%m-%dT00:00Z")

        # Construct Query URL
        query_string = "filename?processEnvs=oe&dataTypes=f1m,m1m&dataStartTime={}&dataEndTime={}".format(
            start_date, end_date
        )
        print(start_date, end_date)
        # Create list of urls for DSCOVR data set
        url = query_url + query_string
        response = requests.get(url)
        filenames = None
        if response.ok:
            response = response.json()
            filenames = response["items"]
        else:
            logging.error("Unable to fetch data from URL: {}".format(url))
            raise Exception("Unable to fetch data from URL: {}".format(url))
        if filenames is None:
            return None
        dflistf1m = []
        dflistm1m = []
        urls = []

        # Get all the data in list of data frames
        for filename in (pbar := tqdm.tqdm(filenames)):
            ind = filename.find("_s")
            yr = filename[ind + 2 : ind + 6]
            mo = filename[ind + 6 : ind + 8]
            downlaod_url = base_url.format(yr, mo, filename)
            urls.append(downlaod_url)
            pbar.set_description(filename)
            logging.info(filename)
            if url_only:
                continue
            if filename.startswith("oe_f1m_dscovr_"):
                dflistf1m.append(
                    _download_dscvr(downlaod_url, data_type="f1m", logger=self.logger)
                )
            elif filename.startswith("oe_m1m_dscovr_"):
                dflistm1m.append(
                    _download_dscvr(downlaod_url, data_type="m1m", logger=self.logger)
                )

        if url_only:
            return urls

        # Merge list of data frames
        merged_df_f1m = pd.concat(dflistf1m)
        merged_df_m1m = pd.concat(dflistm1m)
        filtered_df_f1m = merged_df_f1m[~merged_df_f1m.index.duplicated(keep="first")]
        filtered_df_m1m = merged_df_m1m[~merged_df_m1m.index.duplicated(keep="first")]
        ind = filtered_df_f1m.index.intersection(filtered_df_m1m.index)

        # Make sure that size of two data frames are same
        if filtered_df_f1m.shape[0] != filtered_df_m1m.shape[0]:
            self.logger.warning("Size of two data frames are different.")
        df = filtered_df_f1m.join(filtered_df_m1m, how="inner")

        # Save data in HDF5 file
        tstart = df.index[0].strftime("%Y%m")
        tend = df.index[-1].strftime("%Y%m")
        filename = f"dscovr_{tstart}_{tend}.h5"

        # Convert to cftime to strig so that it can be saved in HDF
        # Without Pickeling
        df.index = [s.isoformat() for s in df.index]
        df.index.name = "Time"

        df.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
        return df

    def get_data(self, observatory: str = "dscovr", **kwargs) -> pd.DataFrame | None:
        """
        This method will download the data from given observatory
        using the internal helper functions.

        Parameters
        ----------
        observatory: str, optional
            The observatory to download data for, by default "dscovr".
        kwargs: dict
            Keywords about the data series based on which the data
            will be downlaoded.
            start_date: datetime.datetime, optional
            Start date of the data range, by default None.
            end_date: datetime.datetime, optional
            End date of the data range, by default None.
            url_only: bool, optional
            Whether or not to download the data from URL, by default False.

        Returns
        -------
        df : pd.DataFrame
            The dataframe containing the data.

        """
        if observatory == "dscovr":
            df = self._dscovr(**kwargs)
            return df
