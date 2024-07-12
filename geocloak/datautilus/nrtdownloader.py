"""

"""


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

    last_data_time = {"dscovr": None}

    def __init__(self, path="data", download: bool = False):
        """Class initiation"""
        self.path = path
        self.download = download
        os.makedirs(path, exist_ok=True)
        self._dscovr_url = (
            "https://www.ngdc.noaa.gov/next-catalogs/rest/dscovr/catalog/"
        )
        self.dscovr_nrt_url = ""
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
            self.last_data_time["dscovr"] = latestTime
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
