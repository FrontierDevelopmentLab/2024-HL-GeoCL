import io

import matplotlib.pyplot as plt
import PIL.Image
import pytorch_lightning as pl
import torch
import wandb
from torchvision.transforms import ToTensor
from utils.helpers import R2
from utils.plot import spherical_plot_forecasting

# ---------------- Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------

# define a function which returns an image as numpy array from figure


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
        (a**2).sum(dim=-1) - (b**2).sum(dim=-1)
    ).mean()


def CompErr(true, pred):
    return ((true - pred) ** 2).mean(dim=(0, 1)).sum() + torch.abs(
        (true**2).sum(dim=-1) - (pred**2).sum(dim=-1)
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

        try:
            self.lossfun = ldict[losskey]
        except Exception:
            self.lossfun = MAE

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, training_batch, batch_idx):
        (
            past_omni,
            future_supermag,
        ) = training_batch

        predictions = self(
            past_omni
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx
        future_supermag = future_supermag[..., target_col].squeeze(1)

        loss = self.lossfun(future_supermag, predictions)

        # sparsity L2
        loss += self.l2reg * torch.norm(predictions, p=2)

        self.log("train_MAE", loss, on_step=False, on_epoch=True, sync_dist=True)
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
        ) = val_batch
        _, coeffs, predictions = self(
            past_omni
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx

        future_supermag = future_supermag[..., target_col].squeeze(1)

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