"""
This python script will prepare omniweb data fro sheath data preparation.
"""

import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.datautilus.downloader import DataDownloader  # noqa: E402

out_dir = "/home/bjha/data/geocloak/formatted_data/OMNI/omniweb_1m"
file_dir = "/home/bjha/data/landingpage/omni/"

a = DataDownloader(outpath=out_dir)

for file in pathlib.Path(file_dir).iterdir():
    if not file.name.endswith("lst"):
        continue
    print(file.name)
    a.omniweb(
        datafile=str(file), fmtfile=os.path.join(file_dir, "omni_min_formatv2.fmt")
    )
