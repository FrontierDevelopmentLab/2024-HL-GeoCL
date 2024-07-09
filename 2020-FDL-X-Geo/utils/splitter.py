import glob
import sys

import h5py
import numpy as np
import pandas as pd
import sklearn.model_selection
from astropy.time import Time

sys.path.append("../")
from utils.data_utils import get_iaga_data_as_list, get_input_data

BUCKETS = 100


def get_sequences(df, length, lag):
    _df = df.copy()
    _df["cluster"] = (_df["seconds"].diff() != 60).cumsum()
    f = (
        _df.groupby(["cluster"])
        .apply(lambda x: x[: len(x) - length - lag]["index"])
        .reset_index(drop=True)
        .values.ravel()
    )  # features (with ttime hist)
    t = (
        _df.groupby(["cluster"])
        .apply(lambda x: x[length + lag :]["index"])
        .reset_index(drop=True)
        .values.ravel()
    )  # targets (with lag)

    # if len(t) > 0:
    #     assert np.max((t-f)) == np.min((t-f)) == (length+lag)

    return list(zip(f, t))


def weimerdatesgetter(base="../data_local/weimer/"):
    """
        Wrapper to get us timesteps of Weimer predictions.
    """
    storm_inds = []
    weimerpaths = sorted(glob.glob(f"{base}*.h5"))
    for fpath in weimerpaths:
        weimer = {}
        with h5py.File(fpath, "r") as f:
            for k in f.keys():
                weimer[k] = f.get(k)[:]
        wtime = weimer["JDTIMES"]
        storm_inds.append(Time(wtime, format="jd").to_value("unix"))
    return storm_inds


def generate_indices(
    base,
    year,
    LENGTH,
    LAG,
    omni_path="../data_local/omni/sw_data.h5",
    weimer_path="../data_local/weimer/",
):
    print(f"loading from path {base} /")

    dates, data, features, _ = get_iaga_data_as_list(
        base, year, tiny=False, load_data=False
    )

    df = pd.DataFrame()
    df["seconds"] = dates
    df["dates"] = pd.to_datetime(df["seconds"], unit="s", errors="coerce")

    df["index"] = range(len(df))
    bucket_size = len(df) // BUCKETS
    df["bucket"] = (((df["index"] % bucket_size) == 0) & (df["index"] > 0)).cumsum()

    # Gets us list of start and end times of storm
    weimertimes = weimerdatesgetter(base=weimer_path)
    weimerbuckets = []
    for dateset in weimertimes:
        start, end = dateset[0], dateset[-1]
        st, ed = (
            df.iloc[np.argmin(np.abs(df["seconds"].values - start))]["bucket"],
            df.iloc[np.argmin(np.abs(df["seconds"].values - end))]["bucket"],
        )
        if (st - ed) != 0:
            weimerbuckets += list(np.arange(st, ed, 1).astype(int))
        else:
            weimerbuckets.append(st)
    # weimerbuckets=np.concatenate(weimerbuckets,axis=0).astype(int)

    N_WEIMER = len(weimerbuckets)
    TRAIN_TEST_SPLIT = [
        a for a in list(np.arange(BUCKETS + 1)) if a not in weimerbuckets
    ]
    train_size = 0.8 + int(N_WEIMER / BUCKETS)

    np.random.seed(0)
    train, test_val = sklearn.model_selection.train_test_split(
        TRAIN_TEST_SPLIT, train_size=train_size
    )
    test, val = sklearn.model_selection.train_test_split(test_val, train_size=0.5)
    weimer = weimerbuckets

    df.loc[df["bucket"].isin(train), "split"] = "train"
    df.loc[df["bucket"].isin(test), "split"] = "test"
    df.loc[df["bucket"].isin(val), "split"] = "val"
    df.loc[df["bucket"].isin(weimer), "split"] = "weimer"

    train_df = df.loc[df["split"] == "train"]
    test_df = df.loc[df["split"] == "test"]
    val_df = df.loc[df["split"] == "val"]
    weimer_df = df.loc[df["split"] == "weimer"]

    # make sure omni and iaga have the same dates
    # test if it maches omni
    for year in pd.unique(df["dates"].dt.year):
        print(f"testing {year}")
        omni = get_input_data(omni_path, year=f"{year}")
        assert len(df.loc[df["dates"].dt.year == year]) == len(omni)
    print("testing done")

    train_idx = get_sequences(train_df, LENGTH, LAG)
    test_idx = get_sequences(test_df, LENGTH, LAG)
    val_idx = get_sequences(val_df, LENGTH, LAG)
    if N_WEIMER:
        weimer_idx = get_sequences(weimer_df, LENGTH, LAG)
    else:
        weimer_idx = None

    return train_idx, test_idx, val_idx, weimer_idx
