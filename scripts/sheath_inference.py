"""
Code for SHEATH inference.
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from astropy.time import Time
from torch.utils.data import DataLoader

# Add geocloak in the python path
sys.path.append(os.path.abspath("../"))

from geocloak.models.sheath import SHEATH_MLP, SHEATHDataLoader  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="SHEATH model inference and plotting.")
    parser.add_argument(
        "--data-dir",
        default="/home/bjha/data/geocloak/formatted_data/sheath_splits",
        help="Directory containing test CSV data",
    )
    parser.add_argument(
        "--scaler-dir",
        default="/home/bjha/sheath_scalar",
        help="Directory containing scaler JSON files",
    )
    parser.add_argument(
        "--checkpoint",
        default="../models/sheath_best_model_checkpoint_1.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        default="sheath_best_hyper.yml",
        help="Path to hyperparameter YAML config",
    )
    parser.add_argument(
        "--test-file",
        default="sheath_test_set.csv",
        help="Name of the test CSV file within data-dir",
    )
    parser.add_argument(
        "--output-dir",
        default="../media/sheath",
        help="Directory to save output plots",
    )
    args = parser.parse_args()

    # Load Model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = SHEATH_MLP(
        input_dim=26,
        hidden_dim=config["hidden_dim"],
        output_dim=14,
        dropout_rate=config["dropout_rate"],
    )
    model.to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataloader = SHEATHDataLoader(
        args.data_dir,
        args.test_file,
        args.scaler_dir,
        is_train=False,
        get_timestamp=True,
    )

    test_loader = DataLoader(dataloader, batch_size=1, shuffle=False)

    xtest = []
    ypred = []
    times = []
    ytrue = []

    for _x, _ytrue, _times in test_loader:
        _ypred = model(_x.to(device))

        _ypred = _ypred.detach().cpu().numpy().flatten()
        xtest.append(_x.numpy())
        ypred.append(_ypred)
        times.append(_times)
        ytrue.append(_ytrue.numpy().flatten())

    times = np.array(times).flatten()
    ftime = Time(times)

    ind = np.argsort(ftime)
    times = times[ind]
    ftime = ftime[ind]
    ytrue = np.array(ytrue)[ind]
    ypred = np.array(ypred)[ind]

    test_intervals = [
        ("2011-08-04", "2011-08-08"),
        ("2017-09-26", "2017-09-29"),
        ("2018-08-13", "2018-08-17"),
        ("2019-08-29", "2019-09-01"),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for j in test_intervals:
        t1, t2 = j
        t1, t2 = Time(t1), Time(t2)
        ind = (ftime > t1) & (ftime < t2)
        fig, ax = plt.subplots(5, 3, figsize=(10, 5))
        for i, _ax in enumerate(ax.flatten()):
            if i > 13:
                break
            _ax.plot(ftime[ind].datetime, ytrue[ind, i], "-", color="tab:blue")
            _ax.plot(ftime[ind].datetime, ypred[ind, i], "-", color="indianred")

        plt.savefig(
            os.path.join(args.output_dir, f"sheath_test_data_{t1.ymdhms[0]}.png"),
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    main()
