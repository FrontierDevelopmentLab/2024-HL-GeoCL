"""
Code for SHEATH inference.
"""

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

# Load Model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model_path = "../models/sheath_best_model_checkpoint_1.pth"

with open("sheath_best_hyper.yml") as f:
    config = yaml.safe_load(f)

model = SHEATH_MLP(
    input_dim=26,
    hidden_dim=config["hidden_dim"],
    output_dim=14,
    dropout_rate=config["dropout_rate"],
)
model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

directory = "/home/bjha/data/geocloak/formatted_data/sheath_splits"
scaler_dir = "/home/bjha/sheath_scalar"

dataloader = SHEATHDataLoader(
    directory, "sheath_test_set.csv", scaler_dir, is_train=False, get_timestamp=True
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

# print(f"{xtest.shape=}")

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

titles = ["Speed", "Density", "Temperature", "Bt", "Bx", "By", "Bz"]

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
        # _ax.set_title(titles[i])

    plt.savefig(f"../media/sheath/sheath_test_data_{t1.ymdhms[0]}.png", dpi=300)
    plt.close()
