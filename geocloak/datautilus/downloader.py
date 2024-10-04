"""
This module provides an API to download various exiting data from the
from multiple observatories and sources using their endpoint APIs or weburl.
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

from geocloak.configs.datainfo import column_names, dscovr_f1m_cols, dscovr_m1m_cols

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
                # Read NetCDF file
                with nc.Dataset("inmemory.nc", memory=gz.read()) as f:
                    idx = nc.num2date(f.variables["time"][:], f.variables["time"].units)

                    # Only Read columns provided in config.datainfo
                    for col in cols:
                        vals = np.asarray(f.variables[col][:])
                        data_list.append(pd.Series(vals, index=idx))
        # Correct format
        df = pd.DataFrame(data_list).transpose()
        # Give column names
        df.columns = cols
        return df
    except Exception as e:
        log.error("Unable to download the data from DSCVR website.{}".format(e))
        return None


def _download_ace(url: str, datatype: str = "mag") -> pd.DataFrame | None:
    """
    A Helper function to download the data from ACE for given link.

    Parameters
    ----------
    url: str
        The url of the downlaodable file.
    datatype: str
        The type of data series to download. Options ["mag", "swepam"]

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the data.

    """
    # Convert to dataframe by skipping rows except one with header
    # Need to be updated if data formating changes
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

    # Reamove redunt column, apeard due to formating
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
    # Clean data and get rid of extra columns
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
        if not provided, the directory will be created with name `data`
        in the current working directory.

    """

    def __init__(self, outpath: str = None):
        self.outpath = outpath
        if self.outpath is None:
            self.outpath = "data"
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath, exist_ok=True)

    def dscovr(
        self,
        start_year: int = None,
        end_year: int = None,
        resample=False,
    ) -> None:
        """
        This method will download the data using DSCOVR weburl ("https://www.ngdc.noaa.gov/dscovr/)
        however it serch the data using NEXT API. The data will be stored in Pandas HDF format in
        each year format. The data availabe is 1m cadence so this menthod also resample the data
        by taking hourly mean using pandas resampling method.

        Parameters
        ----------
        start_year: int, optional
            Start date of the data range, by default None.
        end_year: int, optional
            End date of the data range, by default None.
        resample: bool, optional
            Whether or not to resample the data by hour, by default True.

        Return
        -------
            None
        """
        # Base url to download the data
        base_url = "https://www.ngdc.noaa.gov/dscovr/data/{}/{}/{}"

        # API end point to query the data
        query_url = "https://www.ngdc.noaa.gov/next-catalogs/rest/dscovr/catalog/"

        # Modify start_year and end_year based on input
        start_year = 2015 if start_year is None else start_year
        end_year = datetime.datetime.today().year if end_year is None else end_year

        # Start downloading data yearly
        for year in range(start_year, end_year + 1):

            # Set start and end date of the year
            start_date = datetime.datetime(year, 1, 1).strftime("%Y-%m-%dT00:00Z")
            end_date = datetime.datetime(year, 12, 31).strftime("%Y-%m-%dT23:59Z")

            # Construct Query URL
            query_string = "filename?processEnvs=oe&dataTypes=f1m,m1m&dataStartTime={}&dataEndTime={}".format(
                start_date, end_date
            )
            # Log it
            log.info(f"Downloding data from {start_date} to {end_date}")

            # Create list of urls for DSCOVR data set
            url = query_url + query_string
            response = requests.get(url)

            filenames = None
            if response.ok:
                response = response.json()
                filenames = np.sort(response["items"])
            else:
                log.error("Unable to fetch data from URL: {}".format(url))
                raise Exception("Unable to fetch data from URL: {}".format(url))
            if (filenames is None) or (len(filenames) == 0):
                log.warning("No data found for year {}".format(year))
                continue
            dflistf1m = []
            dflistm1m = []
            urls = []

            # Get all the data in list of data frames
            for filename in (pbar := tqdm.tqdm(filenames)):
                log.info("Downloading data from {}".format(filename))

                # Get year and month to construct download url
                ind = filename.find("_s")
                yr = filename[ind + 2 : ind + 6]
                mo = filename[ind + 6 : ind + 8]
                downlaod_url = base_url.format(yr, mo, filename)
                urls.append(downlaod_url)
                pbar.set_description(filename)

                # Downlaod data based on their type (faradaycup or mag)
                if filename.startswith("oe_f1m_dscovr_"):
                    dflistf1m.append(_download_dscvr(downlaod_url, data_type="f1m"))
                    log.info("Downloaded data from {}".format(downlaod_url))
                elif filename.startswith("oe_m1m_dscovr_"):
                    log.info("Downloaded data from {}".format(downlaod_url))
                    dflistm1m.append(_download_dscvr(downlaod_url, data_type="m1m"))

            # Merge list of data frames if len is not zero
            if len(dflistf1m) == 0:
                log.info("Unable to downlaod any data for year {}".format(year))
                continue
            merged_df_f1m = pd.concat(dflistf1m)
            merged_df_m1m = pd.concat(dflistm1m)

            # Filter data for any repeatation
            filtered_df_f1m = merged_df_f1m[
                ~merged_df_f1m.index.duplicated(keep="first")
            ]
            filtered_df_m1m = merged_df_m1m[
                ~merged_df_m1m.index.duplicated(keep="first")
            ]

            # Make sure that size of two data frames are same
            if filtered_df_f1m.shape[0] != filtered_df_m1m.shape[0]:
                log.warning("Size of two data frames are different.")
            df = filtered_df_f1m.join(filtered_df_m1m, how="inner")

            # Convert to cftime to strig so that it can be saved in HDF
            # Without Pickeling
            df.index = pd.to_datetime([s.isoformat() for s in df.index])
            df.index.name = "Time"
            df.rename(columns=column_names, inplace=True)

            # Resample it to hours if the flag is set.
            if resample:
                df = df.resample("H").mean()

            # Save data in HDF5 file
            t_char = "1h" if resample else "1m"
            filename = f"dscovr_formatted_{t_char}_{year:0>4}.h5"

            # Save data to given directory in HDF format.
            df.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
            log.info(f"Saved DSCOVR data to {os.path.join(self.outpath, filename)}")

    def ace(
        self,
        start_year: int = None,
        end_year: int = None,
        resample: bool = False,
    ) -> pd.DataFrame | None:
        """
        This method will download the exiting ACE data in a given time range.

        Parameters
        ----------
        start_year: datetime.datetime, optional
            Start date of the data range, by default None.
        end_year: datetime.datetime, optional
            End date of the data range, by default None.
        resample: bool, optional
            Whether or not to resample the data by hour, by default True.

        Returns
        -------
        None

        """
        # Download url
        base_url = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"

        # Convert date time to datetime object
        start_year = 2015 if start_year is None else start_year
        end_year = datetime.datetime.today().year if end_year is None else end_year

        # Query weburl for available datasets
        try:
            res = urllib.request.urlopen(base_url)
        except FileNotFoundError:
            log.warning("Unable to download ACE data from URL: {}".format(base_url))
            raise Exception("Unable to download ACE data from URL: {}".format(base_url))

        # Get beautiful
        soup = BeautifulSoup(res.read(), "html.parser")

        for year in range(start_year, end_year + 1):

            # Get the format string
            mag = re.compile("^{}[0-9]+_ace_mag_1m.txt".format(year))
            swepam = re.compile("^{}[0-9]+_ace_swepam_1m.txt".format(year))

            # Okay, its time to get the links and list it
            mag_list = []
            swepam_list = []
            for a in soup.find_all("a"):
                if mag.match(a.text):
                    mag_list.append(a.text)
                if swepam.match(a.text):
                    swepam_list.append(a.text)

            # Downlaod data and append them in the list
            dfmag = []
            dfswepam = []
            with tqdm.tqdm(total=len(mag_list)) as pbar:
                for _mag, _swepam in zip(mag_list, swepam_list):
                    pbar.set_description(_mag)
                    pbar.update()
                    dfmag.append(_download_ace(base_url + _mag, datatype="mag"))
                    dfswepam.append(
                        _download_ace(base_url + _swepam, datatype="swepam")
                    )

            # Concate the dataframes if no of data frame is nonzero
            if len(dfmag) == 0:
                continue
            merged_df_mag = pd.concat(dfmag)
            merged_df_swepam = pd.concat(dfswepam)
            filtered_df_mag = merged_df_mag[
                ~merged_df_mag.index.duplicated(keep="first")
            ]
            filtered_df_swepam = merged_df_swepam[
                ~merged_df_swepam.index.duplicated(keep="first")
            ]

            # Make sure that size of two data frames are same
            if filtered_df_mag.shape[0] != filtered_df_swepam.shape[0]:
                log.warning("Size of two data frames are different.")
            df = filtered_df_swepam.join(filtered_df_mag, how="inner")

            # Resample it to hours if the flag is set.
            if resample:
                df = df.resample("H").mean()

            # Index it and rearange the columns for consitency, replace
            # fill values with Nan.
            df.index.name = "Time"
            df = df[["Speed", "Density", "Temperature", "Bt", "Bx", "By", "Bz"]].copy()
            df.replace(
                {
                    -9999.9: np.nan,
                    -100000.0: np.nan,
                    -999.9: np.nan,
                    9999999.0: np.nan,
                    9.999: np.nan,
                    999.99: np.nan,
                    9999.99: np.nan,
                    99999.9: np.nan,
                },
                inplace=True,
            )
            # Save data in HDF5 file
            t_char = "1h" if resample else "1m"
            filename = f"ace_formatted_{t_char}_{year}.h5"

            # Hurrey, save the data.
            df.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
            log.info(f"Saved ACE data to {os.path.join(self.outpath, filename)}")

    def omniweb(
        self, datafile: str | list, fmtfile: str, **kwargs
    ) -> pd.DataFrame | None:
        """To transform omni data into formated data frame and save it
         as hdf5 for each year file.

        Parameters
        ----------
        datafile: str | list
            The path to the data file.
        fmtfile: str
            The path to the format file.

        Returns
        ----------
        data: pd.DataFrame | None
            The data frame containg all the data from the given file.
        """
        # New column name dictionary to update names
        cols = {
            "Scalar B, nT": "Bt",
            "BX, nT (GSE, GSM)": "Bx",
            "BY, nT (GSM)": "By",
            "BZ, nT (GSM)": "Bz",
            "SW Plasma Temperature, K": "Temperature",
            "SW Proton Density, N/cm^3": "Density",
            "SW Plasma Speed, km/s": "Speed",
        }
        # Read formate file to get the list of available variables.

        fmt = pd.read_fwf(fmtfile, sep=" ", engine="python", header=1).values[:, 0]

        heads = [" ".join(t.split(" ")[1:]) for t in fmt]

        # Read data from lst file
        data = pd.read_csv(
            datafile,
            sep="\s{1,}",  # noqa: W605
            engine="python",
            header=None,
            names=heads,
        )

        # Convert day of year to ISO datetime format
        data["Time"] = data.apply(
            lambda row: datetime.datetime(
                int(row.YEAR), 1, 1, int(row.Hour), int(row.Minute)
            )
            + datetime.timedelta(row.DOY - 1),
            axis=1,
        )

        # Ohh, get rid of some of the coumns
        data.drop(["YEAR", "DOY", "Hour"], inplace=True, axis=1)
        data = data[["Time"] + [col for col in data.columns if col != "Time"]]
        data.set_index("Time", inplace=True)

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

        # Format the data as other data sources (ACE, DSCOVR)
        data.index = pd.to_datetime(data.index)
        data = data[cols.keys()]
        data.rename(columns=cols, inplace=True)
        data = data[["Speed", "Density", "Temperature", "Bt", "Bx", "By", "Bz"]]

        # Save yearly data to hdf panda data frames
        for year in (pbar := tqdm.tqdm(np.unique(data.index.year))):
            pbar.set_description(str(year))
            _data = data.loc[str(year)].copy()
            # Save data in HDF5 file
            filename = f"omniweb_formatted_1m_{year}.h5"
            _data.to_hdf(os.path.join(self.outpath, filename), key="data", mode="w")
            log.info(f"Saved OMNI data to {os.path.join(self.outpath, filename)}")

        return data
