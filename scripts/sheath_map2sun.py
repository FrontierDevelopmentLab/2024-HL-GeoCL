"""
This python script will transform the omni time stamp into sdo time stamp.
Basically traceback the solar wind plasma back to the Sun through the parker Spiral.
"""

import sys
import os
import pathlib
import pandas as pd
import tqdm as tq

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.preprocess.backtrack import ballistic, HUX


OMNI_DIR = "/home/bjha/data/geocloak/formatted_data/OMNI/omniweb_1m"
omnipath = pathlib.Path(OMNI_DIR)


def map2sun(year: int) -> None:
    filepath = omnipath / f"omniweb_formatted_1m_{year}.h5"
    _df2 = pd.read_hdf(filepath)

    filepath = omnipath / f"omniweb_formatted_1m_{year-1}.h5"
    _df1 = pd.read_hdf(filepath)
    df = pd.concat([_df1, _df2])

    timedata = {"Time": [], "Ballistic": [], "HUX": []}
    print(df.shape)

    for i in tq.tqdm(range(_df1.shape[0], df.shape[0])):
        time_ballistic = ballistic(df.index[i], df.Speed.values[i])
        dindex = int(27 * 24 * 60)
        time_hux = HUX(
            df.index[i - dindex : i + 1 : 60], df.Speed.values[i - dindex : i + 1 : 60]
        )

        timedata["Ballistic"].append(time_ballistic)
        timedata["HUX"].append(time_hux)
        timedata["Time"].append(df.index[i])
        if i == 200:
            break

    pd.DataFrame(timedata).to_csv(omnipath / f"omniweb_map2sun_1m_{year}.csv")


if __name__ == "__main__":
    map2sun(2015)
