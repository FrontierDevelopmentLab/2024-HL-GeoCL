import io

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from dataloader import basis_matrix
from matplotlib import cycler
from torchvision.transforms import ToTensor

# ---------------- Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------


class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        # self.vmin = vmin # minimum value
        self.mid = mid  # middle value
        # self.vmax = vmax # maximum value
        self.s1 = s1
        self.s2 = s2
        f = (
            lambda x, zero, vmax, s: np.abs((x - zero) / (vmax - zero)) ** (1.0 / s)
            * 0.5
        )
        self.g = (
            lambda x, zero, vmin, vmax, s1, s2: f(x, zero, vmax, s1) * (x >= zero)
            - f(x, zero, vmin, s2) * (x < zero)
            + 0.5
        )
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid, self._vmin, self._vmax, self.s1, self.s2)
        return np.ma.masked_array(r)


def spherical_plot_forecasting(
    nmax, coeffs, predictions, target, mlt, mcolat, mean, std
):
    plt.style.use("default")
    plt.rcParams.update(
        {
            "lines.linewidth": 1.0,
            "axes.grid": False,
            "grid.linestyle": ":",
            "axes.grid.axis": "both",
            "axes.prop_cycle": cycler(
                "color",
                ["0071bc", "d85218", "ecb01f", "7d2e8d", "76ab2f", "4cbded", "a1132e"],
            ),
            "xtick.top": True,
            "xtick.minor.size": 0,
            "xtick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.right": True,
            "ytick.minor.size": 0,
            "ytick.direction": "in",
            "ytick.minor.visible": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "white",
            "legend.fancybox": False,
            "figure.figsize": (12, 12),
            "figure.autolayout": False,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.01,
            "savefig.transparent": True,
        }
    )

    shape_spherical = (45, 180)  # colat, lon
    _phi_spherical = (
        (np.arange(shape_spherical[0]) + 0.5) / shape_spherical[0] * np.pi / 4
    )  # colat
    _theta_spherical = (
        (np.arange(shape_spherical[1]) + 0.5) / shape_spherical[1] * 2.0 * np.pi
    )  # lon

    grid_theta_spherical, grid_phi_spherical = np.meshgrid(
        _theta_spherical, _phi_spherical
    )

    basis_grid = basis_matrix(
        nmax,
        grid_theta_spherical,
        grid_phi_spherical,
    )
    basis_grid = torch.Tensor(basis_grid).double().squeeze(0).to(device)

    cm.get_cmap("viridis")

    grid_predictions = (basis_grid @ coeffs.T).detach().cpu().numpy()
    grid_predictions = grid_predictions.reshape(-1, *_theta_spherical.shape)

    cmap = "PuOr_r"

    maxval = 300
    minval = -300
    norm = SqueezedNorm(vmin=minval, vmax=maxval, mid=0, s1=2, s2=2)

    fig, ax = plt.subplots(ncols=4, subplot_kw={"projection": "polar"})

    ax[0].set_theta_offset(-np.pi / 2)
    c = ax[0].scatter(mlt, mcolat, c=target, cmap=cmap, norm=norm)
    ax[0].set_title("Target")

    ax[1].set_theta_offset(-np.pi / 2)
    c = ax[1].scatter(mlt, mcolat, c=predictions, cmap=cmap, norm=norm)
    ax[1].set_title("Predictions")

    ax[2].set_theta_offset(-np.pi / 2)
    ax[2].pcolormesh(
        grid_theta_spherical,
        grid_phi_spherical,
        grid_predictions * std + mean,  # un-standardize
        cmap=cmap,
        shading="auto",
        norm=norm,
    )
    ax[2].set_title("Prediction (SpH)")
    ax[2].scatter(
        mlt, mcolat, c=predictions, cmap=cmap, norm=norm, s=20, edgecolors="k"
    )

    ax[3].set_axis_off()
    cb = fig.colorbar(c, ax=ax[3], shrink=0.2, location="left")
    cb.set_label("dB [nT]", fontsize=14, labelpad=-70)

    plt.subplots_adjust(wspace=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    return ToTensor()(image)
