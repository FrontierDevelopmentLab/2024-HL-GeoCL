import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils.helpers import R2
from utils.plot import spherical_plot_forecasting

# ---------------- Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    return ToTensor()(image)


def SumSE(a, b):
    return ((a - b) ** 2).sum()


def MSE(a, b):
    return ((a - b) ** 2).mean()


def MAE(a, b):
    return (torch.abs(a - b)).mean()


def weighted_MAE(a, b, weight=1):
    return (torch.abs(a - b) * weight).mean()


def MaxSqEr(true, pred):
    return torch.sum(((true - pred) ** 2).mean(dim=(0, 1)), dim=-1)[0] + MSE(true, pred)


def SqSqEr(true, pred):
    return ((true - pred) ** 4).mean()


def MAE_BH(a, b):
    return (torch.abs(a - b)).mean() + torch.abs(
        (a ** 2).sum(dim=-1) - (b ** 2).sum(dim=-1)
    ).mean()


def CompErr(true, pred):
    return ((true - pred) ** 2).mean(dim=(0, 1)).sum() + torch.abs(
        (true ** 2).sum(dim=-1) - (pred ** 2).sum(dim=-1)
    ).mean()


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        ldict = {
            "MSE": MSE,
            "MAE": MAE,
            "SumSE": SumSE,
            "MaxSqEr": MaxSqEr,
            "SqSqEr": SqSqEr,
            "CompErr": CompErr,
            "MAE_BH": MAE_BH,
        }
        ldict_weighted = {"MAE": weighted_MAE}
        self.lr = kwargs.pop("learning_rate", 1e-4)
        self.l2reg = kwargs.pop("l2reg", 1e-4)
        losskey = kwargs.pop("loss", None)
        self.weighted_regression = kwargs.pop("weighted_regression", None)
        self.stn_reg = kwargs.pop("stn_reg", False)
        self.imbalanced_regression_weight = kwargs.pop("imbalanced_regression_weight")
        self.station_regularization_weight = kwargs.pop("station_regularization_weight")
        self.stn_reg_dmp = kwargs.pop("stn_reg_dmp")

        try:
            if self.weighted_regression or self.stn_reg:
                self.lossfun = ldict_weighted[losskey]
            else:
                self.lossfun = ldict[losskey]
        except:
            self.lossfun = MSE

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, training_batch, batch_idx):
        (
            past_omni,
            past_supermag,
            future_supermag,
            future_supermag_reg,
            past_dates,
            future_dates,
            (phi, theta),
            weight_dbe,
            weight_dbn,
        ) = training_batch

        _, coeffs, predictions = self(
            past_omni, past_supermag, phi, theta, past_dates, future_dates
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx
        future_supermag = future_supermag[..., target_col].squeeze(1)

        if self.weighted_regression and self.stn_reg:
            imbalanced_regression_loss = self.lossfun(
                future_supermag,
                predictions,
                torch.stack((weight_dbe, weight_dbn), dim=-1),
            )
            future_supermag_reg = 1.0 - self.stn_reg_dmp * (1.0 - future_supermag_reg)
            stn_reg_loss = self.lossfun(
                future_supermag, predictions, future_supermag_reg
            )
            loss = self.imbalanced_regression_weight * imbalanced_regression_loss
            +self.station_regularization_weight * stn_reg_loss

        elif self.weighted_regression:
            loss = self.lossfun(
                future_supermag,
                predictions,
                torch.stack((weight_dbe, weight_dbn), dim=-1),
            )

        elif self.stn_reg:
            future_supermag_reg = 1.0 - self.stn_reg_dmp * (1.0 - future_supermag_reg)
            loss = self.lossfun(future_supermag, predictions, future_supermag_reg)

        else:
            loss = self.lossfun(future_supermag, predictions)

        # sparsity L2
        loss += self.l2reg * torch.norm(coeffs, p=2)

        self.log("train_MSE", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train_r2",
            R2(future_supermag, predictions).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "train_dbe_MSE",
            MSE(future_supermag[..., [0]], predictions[..., [0]]).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_dbe_r2",
            R2(future_supermag[..., [0]], predictions[..., [0]]).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "train_dbn_MSE",
            MSE(future_supermag[..., [1]], predictions[..., [1]]).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_dbn_r2",
            R2(future_supermag[..., [1]], predictions[..., [1]]).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            past_omni,
            past_supermag,
            future_supermag,
            future_supermag_reg,
            past_dates,
            future_dates,
            (mlt, mcolat),
            weight_dbe,
            weight_dbn,
        ) = val_batch
        _, coeffs, predictions = self(
            past_omni, past_supermag, mlt, mcolat, past_dates, future_dates
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx

        future_supermag = future_supermag[..., target_col].squeeze(1)

        # unstandarize predictions and futures
        # unscaled_future_supermag = self.scaler['supermag'].inverse_transform(future_supermag.cpu().numpy())
        # unscaled_predictions = self.scaler['supermag'].inverse_transform(predictions.cpu().numpy())

        # loss = ((future_supermag - predictions) ** 2).mean()
        loss = self.lossfun(future_supermag, predictions)

        self.log(
            "val_R2",
            R2(future_supermag, predictions).mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_MSE",
            MSE(future_supermag, predictions),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            # hack: need to find a callback way
            predictions = []
            coeffs = []
            targets = []
            mlt_all = []
            mcolat_all = []
            for (
                past_omni,
                past_supermag,
                future_supermag,
                future_supermag_reg,
                past_dates,
                future_dates,
                (mlt, mcolat),
                weight_dbe,
                weight_dbn,
            ) in self.wiemer_data:
                past_omni = past_omni.to(device)
                past_supermag = past_supermag.to(device)
                mlt = mlt.to(device)
                mcolat = mcolat.to(device)
                past_dates = past_dates.to(device)
                future_dates = future_dates.to(device)

                _, _coeffs, pred = self(
                    past_omni, past_supermag, mlt, mcolat, past_dates, future_dates
                )

                predictions.append(pred.to(device))
                coeffs.append(_coeffs.to(device))
                targets.append(future_supermag[..., target_col].to(device))
                mlt_all.append(mlt.to(device))
                mcolat_all.append(mcolat.to(device))
            predictions = torch.cat(predictions).detach()
            coeffs = torch.cat(coeffs).detach()
            targets = torch.cat(targets).detach().squeeze(1)
            mlt_all = torch.cat(mlt_all).detach().squeeze(1)
            mcolat_all = torch.cat(mcolat_all).detach().squeeze(1)

            predictions[torch.isnan(predictions)] = 0
            targets[torch.isnan(targets)] = 0

            _mean, _std = self.scaler["supermag"]
            predictions = predictions * torch.Tensor(_std).to(device) + torch.Tensor(
                _mean
            ).to(device)
            targets = targets * torch.Tensor(_std).to(device) + torch.Tensor(_mean).to(
                device
            )

            self.log(
                "wiemer_R2",
                R2(targets, predictions).mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "wiemer_MSE",
                ((targets - predictions) ** 2).mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            if self.logger is not None:

                fig, ax = plt.subplots()

                ax.scatter(
                    targets.cpu().numpy().ravel(), predictions.cpu().numpy().ravel()
                )
                self.logger.experiment.log(
                    {
                        "wiemer_scatter": [
                            wandb.Image(get_img_from_fig(fig), caption="val_scatter")
                        ]
                    }
                )
                nice_idx = [0]

                plt.close()

                # dbe_nez
                pred_sphere = spherical_plot_forecasting(
                    self.nmax,
                    coeffs[nice_idx][..., 0].squeeze(0),
                    predictions[nice_idx][..., 0].squeeze(0).detach().cpu(),
                    targets[nice_idx][..., 0].squeeze(0).detach().cpu(),
                    mlt_all[nice_idx].squeeze().detach().cpu(),
                    mcolat_all[nice_idx].squeeze().detach().cpu(),
                    _mean[0],
                    _std[0],
                )

                self.logger.experiment.log(
                    {"dbe_nez": [wandb.Image(pred_sphere, caption="pred_sphere")]}
                )

                plt.close()

                # dbn_nez
                pred_sphere = spherical_plot_forecasting(
                    self.nmax,
                    coeffs[nice_idx][..., 1],
                    predictions[nice_idx][..., 1].detach().cpu(),
                    targets[nice_idx][..., 1].detach().cpu(),
                    mlt_all[nice_idx].detach().cpu(),
                    mcolat_all[nice_idx].detach().cpu(),
                    _mean[1],
                    _std[1],
                )
                self.logger.experiment.log(
                    {"dbn_nez": [wandb.Image(pred_sphere, caption="pred_sphere")]}
                )

                plt.close()
