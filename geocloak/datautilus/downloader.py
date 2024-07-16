"""
This module provides an API to download various exiting data from the
from multiple observatories and sources using their endpoint APIs.
"""

import datetime
import gzip
import logging
import os
import re
import urllib
from io import BytesIO

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup
from geocloak.configs.datainfo import dscovr_f1m_cols, dscovr_m1m_cols

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
    except FileNotFoundError:
        log.error("Unable to download the data from DSCVR website.")
        return None


def _download_ace(url: str, datatype: str = "mag") -> pd.DataFrame | None:
    """
    A Helper function to download the data from ACE for given link.

    Parameters
    ----------
    url: str
        The url of the downlaodable file.
    datatype: str
        The type of data series to download.

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the data.

    """
    # Convert to dataframe by skipping rows except one with header
    if datatype == "mag":
        indices = list(range(20))
        indices.remove(18)
    elif datatype == "swepam":
        indices = list(range(18))
        indices.remove(16)

    # Read data by skiping header rows
    try:
        df = pd.read_table(url, sep=r"\s+", skiprows=indices)
    except FileNotFoundError:
        log.error("Unable to download the data from ACE.")
        df = None

    names = df.columns[1:]
    df = df.iloc[:, :-1]
    df.columns = names

    # Formate time information
    df["Time"] = df.apply(
        lambda x: pd.to_datetime(
            f"{int(x.YR)}{int(x.MO):0>2}{int(x.DA):0>2}{int(x.HHMM):0>4}",
            format="%Y%m%d%H%M",
        ),
        axis=1,
    )
    rcols = ["YR", "MO", "DA", "HHMM", "Day", "Day.1", "S"]
    log.info(f"Removing {rcols} from the data frame.")
    df.drop(["YR", "MO", "DA", "HHMM", "Day", "Day.1", "S"], inplace=True, axis=1)
    df.set_index("Time", inplace=True)
    return df


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

    def dscovr(
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
        log.info(f"Downloding data from {start_date} to {end_date}")

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
            log.info("Downloading data from {}".format(filename))

            yr = filename[ind + 2 : ind + 6]
            mo = filename[ind + 6 : ind + 8]
            downlaod_url = base_url.format(yr, mo, filename)
            urls.append(downlaod_url)
            pbar.set_description(filename)
            if url_only:
                continue
            if filename.startswith("oe_f1m_dscovr_"):
                dflistf1m.append(_download_dscvr(downlaod_url, data_type="f1m"))
            elif filename.startswith("oe_m1m_dscovr_"):
                dflistm1m.append(_download_dscvr(downlaod_url, data_type="m1m"))

        if url_only:
            log.warning("Only URL is set, returning only downloadable url links.")
            return urls

        # Merge list of data frames
        merged_df_f1m = pd.concat(dflistf1m)
        merged_df_m1m = pd.concat(dflistm1m)
        filtered_df_f1m = merged_df_f1m[~merged_df_f1m.index.duplicated(keep="first")]
        filtered_df_m1m = merged_df_m1m[~merged_df_m1m.index.duplicated(keep="first")]

        # Make sure that size of two data frames are same
        if filtered_df_f1m.shape[0] != filtered_df_m1m.shape[0]:
            log.warning("Size of two data frames are different.")
        df = filtered_df_f1m.join(filtered_df_m1m, how="inner")

        # Save data in HDF5 file
        tstart = df.index[0].strftime("%Y%m%d")
        tend = df.index[-1].strftime("%Y%m%d")
        filename = f"dscovr_{tstart}_{tend}.h5"

        # Convert to cftime to strig so that it can be saved in HDF
        # Without Pickeling
        df.index = [s.isoformat() for s in df.index]
        df.index.name = "Time"

        df.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
        log.info(f"Saved DSCOVR data to {os.path.join(self.outpath, filename)}")
        return df

    def ace(
        self,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        url_only: bool = False,
    ) -> pd.DataFrame | None:
        """
        This method will download the exiting ACE data in a given time range.

        Parameters
        ----------
        start_date: datetime.datetime, optional
            Start date of the data range, by default None.
        end_date: datetime.datetime, optional
            End date of the data range, by default None.
        url_only: bool, optional
            Whether or not to download the data from URL, by default False.

        Returns
        -------

        """
        base_url = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
        mag = re.compile("^[0-9]+_ace_mag_1m.txt")
        swepam = re.compile("^[0-9]+_ace_swepam_1m.txt")

        # Convert date time to datetime object
        if start_date is None:
            start_date = datetime.datetime(2015, 1, 1)
        if end_date is None:
            end_date = datetime.datetime.now()

        try:
            res = urllib.request.urlopen(base_url)
        except FileNotFoundError:
            log.warning("Unable to download ACE data from URL: {}".format(base_url))
            raise Exception("Unable to download ACE data from URL: {}".format(base_url))
        soup = BeautifulSoup(res.read(), "html.parser")
        mag_list = []
        swepam_list = []

        for a in soup.find_all("a"):
            if mag.match(a.text):
                times = datetime.datetime.strptime(a.text[0:8], "%Y%m%d")
                if (times >= start_date) and (times <= end_date):
                    mag_list.append(a.text)
            if swepam.match(a.text):
                times = datetime.datetime.strptime(a.text[0:8], "%Y%m%d")
                if (times >= start_date) and (times <= end_date):
                    swepam_list.append(a.text)

        dfmag = []
        dfswepam = []
        with tqdm.tqdm(total=len(mag_list)) as pbar:
            for _mag, _swepam in zip(mag_list, swepam_list):
                pbar.set_description(_mag)
                pbar.update()
                dfmag.append(_download_ace(base_url + _mag, datatype="mag"))
                dfswepam.append(_download_ace(base_url + _swepam, datatype="swepam"))

        merged_df_mag = pd.concat(dfmag)
        merged_df_swepam = pd.concat(dfswepam)
        filtered_df_mag = merged_df_mag[~merged_df_mag.index.duplicated(keep="first")]
        filtered_df_swepam = merged_df_swepam[
            ~merged_df_swepam.index.duplicated(keep="first")
        ]

        # Make sure that size of two data frames are same
        if filtered_df_mag.shape[0] != filtered_df_swepam.shape[0]:
            log.warning("Size of two data frames are different.")
        df = filtered_df_mag.join(filtered_df_swepam, how="inner")

        # Save data in HDF5 file
        tstart = df.index[0].strftime("%Y%m%d")
        tend = df.index[-1].strftime("%Y%m%d")
        filename = f"ace_{tstart}_{tend}.h5"

        # Convert to cftime to strig so that it can be saved in HDF
        # Without Pickeling
        df.index = [s.isoformat() for s in df.index]
        df.index.name = "Time"

        df.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
        log.info(f"Saved ACE data to {os.path.join(self.outpath, filename)}")
        return df

    def omniweb(self, datafile: str, fmtfile: str, **kwargs) -> pd.DataFrame | None:
        """To transform omni data into formated data frame and save it as hdf5 file.

        Parameters
        ----------
        datafile: str
            The path to the data file.
        fmtfile: str
            The path to the format file.

        Returns
        ----------

        """
        fmt = pd.read_fwf(fmtfile, sep=" ", engine="python", header=1).values[:, 0]

        heads = [" ".join(t.split(" ")[1:]) for t in fmt]
        data = pd.read_csv(
            datafile, sep="\s{1,}", engine="python", header=None, names=heads
        )
        data["Date"] = data.apply(
            lambda row: datetime.datetime(int(row.YEAR), 1, 1, int(row.Hour), 0)
            + datetime.timedelta(row.DOY - 1),
            axis=1,
        )
        data.drop(["YEAR", "DOY", "Hour"], inplace=True, axis=1)
        data = data[["Date"] + [col for col in data.columns if col != "Date"]]
        data.set_index("Date", inplace=True)

        # Convert all the 9999s to nan
        log.info("Replacing default fill values with NaN")
        data.replace(
            {
                9999.0: np.nan,
                999.9: np.nan,
                9999999.0: np.nan,
                9.999: np.nan,
                999.99: np.nan,
                9999.99: np.nan,
                99999.9: np.nan,
            },
            inplace=True,
        )

        # Save data in HDF5 file
        tstart = data.index[0].strftime("%Y%m%d")
        tend = data.index[-1].strftime("%Y%m%d")
        filename = f"omniweb_{tstart}_{tend}.h5"
        data.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
        log.info(f"Saved OMNI data to {os.path.join(self.outpath, filename)}")

        return data
