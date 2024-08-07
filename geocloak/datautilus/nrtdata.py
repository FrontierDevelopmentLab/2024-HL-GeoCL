"""
Python Module to download the NRT data from NOAA website.
"""

import logging
import os
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from geocloak.configs.datainfo import dscovr_f1m_cols, dscovr_m1m_cols
from astropy.io import fits


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

    def nrtACE(self, recent_timestamp=None):
        """To get Real Time Solar Wind Data.

        Parameters
        ----------
        recent_timestamp: str, optional
            The recent timestamp of the data. Defaults to None.

        Returns
        -------
        _df: pd.DataFrame | None
            Dataframe containing the NRT ACE and DSCOVR data.
        """
        data = self.__rtsw()
        # if no data found
        if data is None:
            log.warning("No Real Time Data is available.")
            return None
        _df1 = (data[0][data[0].source == "ACE"]).copy()
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = (data[1][data[1].source == "ACE"]).copy()
        _df2.drop(["source"], axis=1, inplace=True)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"

        # If recent timestamp is provided
        if recent_timestamp is not None:
            recent_timestamp = pd.to_datetime(recent_timestamp)
            _df = _df.iloc[pd.to_datetime(_df.index) > recent_timestamp]

        return _df

    def nrtDSCOVER(self, recent_timestamp=None):
        """To get Real Time Solar Wind Data from DSCOVR.

        Parameters
        ----------
        recent_timestamp: str, optional
            The recent timestamp of the data. Defaults to None.

        Returns
        -------
        _df: pd.DataFrame | None
            Dataframe containing the NRT ACE and DSCOVR data.
        """
        data = self.__rtsw()
        if data is None:
            log.warning("No Real Time Data is available.")
            return None

        _df1 = (data[0][data[0].source == "DSCOVR"]).copy()
        _df1.drop(["source"], axis=1, inplace=True)

        _df2 = (data[1][data[1].source == "DSCOVR"]).copy()
        _df2.drop(["source"], axis=1, inplace=True)
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"

        # If recent timestamp is provided
        if recent_timestamp is not None:
            recent_timestamp = pd.to_datetime(recent_timestamp)
            _df = _df.iloc[pd.to_datetime(_df.index) > recent_timestamp]
        return _df

    def nrtbothAD(self, recent_timestamp=None):
        """Property to get combined data from both the observatories.

        Parameters
        ----------
        recent_timestamp: str, optional
            The recent timestamp of the data. Defaults to None.

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
        _df2 = (data[1].drop(["source"], axis=1)).copy()
        _df = _df1.join(_df2, how="inner")
        _df.index.name = "Time"

        # If recent timestamp is provided
        if recent_timestamp is not None:
            recent_timestamp = pd.to_datetime(recent_timestamp)
            _df = _df.iloc[pd.to_datetime(_df.index) > recent_timestamp]

        return _df

    def nrtSDO(self, rootdir: str, recent_timestamp: str = None):
        """
        Method to get Near Real Time data from SDO. If the recent_timestamp
        is given it will tray to get the nearest one for given date

        Parameters
        ----------
        rootdir: str
            The directory where the SDO data is stored.
        recent_timestamp: str, optional
            The recent timestamp of the data. Defaults to None.

        Returns
        -------
        outdata: dict
            A dictionary containing the SDO data and time.

        """

        # Whether the given path exist or not
        sdopath = pathlib.Path(rootdir)
        if not sdopath.exists():
            raise FileNotFoundError(f"{rootdir} doesnot exist.")

        # Find the time and data in HMI closest to recent_time
        if recent_timestamp is not None:
            recent_timestamp = pd.to_datetime(recent_timestamp)
            _path = recent_timestamp.strftime("%Y/%m/%d/")
            _path = sdopath / "HMI" / _path

            # No data for given date
            if not _path.exists():
                print("No HMI data found for the given date.")
                return None

            # Get list of timestamp
            lastdata = np.sort([l for l in _path.glob("*") if l.is_dir()])
            times = [
                datetime.strptime(s.name, "hmi.b_720s.%Y%m%d_%H%M%S_TAI")
                for s in lastdata
            ]

            # Get the nearest one
            ind = np.argmin(np.abs(recent_timestamp - pd.to_datetime(times)))
            lasthmitime = times[ind]
            lastdata = lastdata[ind]

        else:
            # Get the last data for given year and month
            year = np.sort([l for l in sdopath.glob("HMI/*") if l.is_dir()])[-1]
            month = np.sort([l for l in year.glob("*") if l.is_dir()])[-1]
            day = np.sort([l for l in month.glob("*") if l.is_dir()])[-1]
            lastdata = np.sort([l for l in day.glob("*") if l.is_dir()])[-1]
            lasthmitime = datetime.strptime(
                lastdata.name, "hmi.b_720s.%Y%m%d_%H%M%S_TAI"
            )

        # Get the list of files for HMI and AIA
        filelist = {}
        sdodata = {}
        magcompo = ("Bx", "By", "Bz")
        sdotimes = {
            "Bx": lasthmitime.isoformat(),
            "By": lasthmitime.isoformat(),
            "Bz": lasthmitime.isoformat(),
        }

        # Update list of file for HMI
        for mag in magcompo:
            filelist[mag] = str(list(lastdata.glob(f"*{mag}.fits"))[-1])

        # Get AIA data based on the lastdata
        aiapath = sdopath / "AIA" / lasthmitime.strftime("%Y/%m/%d/H%H00")

        # List existing channels
        waves = tuple(
            {
                l.stem.split("_")[-1]
                for l in aiapath.glob("*")
                if l.name.endswith("fits")
            }
        )

        # Get nearest AIA data for each AIA channel
        for channel in waves:
            filenames = list(aiapath.glob(f"*{channel}.fits"))
            times = [
                datetime.strptime(s.name, f"AIA%Y%m%d_%H%M%S_{channel}.fits")
                for s in filenames
                if s.name.endswith(f"{channel}.fits")
            ]
            ind = np.argmin(np.abs(pd.to_datetime(times) - pd.to_datetime(lasthmitime)))

            # Make the keyword consitent with SDO preproces data
            filelist[f"{int(channel)}A"] = str(filenames[ind])
            sdotimes[f"{int(channel)}A"] = times[ind].isoformat()

        # Read the data
        for channel in filelist:
            with fits.open(filelist[channel]) as hdu:
                sdodata[channel] = hdu[0].data

        outdata = {"Time": sdotimes, "Data": sdodata}

        return outdata
