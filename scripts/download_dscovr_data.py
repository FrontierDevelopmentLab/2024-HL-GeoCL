"""
This module will downlaod the ACE and DISCOVER data from their respective websites.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.datautilus.downloader import DataDownloader  # noqa: E402

START_YEAR = 2015
END_YEAR = None
out_dir = "/home/bjha/data/geocloak/formatted_data/DSCOVR/dscovr_1m"

a = DataDownloader(outpath=out_dir)
a.dscovr(start_year=START_YEAR, end_year=END_YEAR)
