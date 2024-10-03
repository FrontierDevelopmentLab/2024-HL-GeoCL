"""
This module will downlaod the ACE and DISCOVER data from their respective websites.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.datautilus.downloader import DataDownloader

START_YEAR = 2001
END_YEAR = None
out_dir = "/home/bjha/data/geocloak/formatted_data/ACE/ace_1m"

a = DataDownloader(outpath=out_dir)
a.ace(start_year=START_YEAR, end_year=END_YEAR)
