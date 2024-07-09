import os.path
import pickle

import numpy as np
import pytorch_lightning as pl
import torch.optim
from dataloader import ShpericalHarmonicsDatasetPreprocessed
from experiment import Experiment
from models.geoeffectivenet import NeuralRNNWiemer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

md = {"NeuralRNNWiemer": NeuralRNNWiemer}

config_path = "experiment.yaml"

# ----- Data loading also depends on the sweep parameters.
# ----- Hence this process will be repeated per training cycle.


def train(config):
    past_omni_length = config["past_omni_length"]
    omni_resolution = config["omni_resolution"]
    nmax = config["nmax"]
    targets = ["dbe_nez", "dbn_nez"]  # config.targets
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    l2reg = config["l2reg"]
    max_epochs = config["epochs"]
    n_hidden = config["n_hidden"]
    dropout_prob = config["dropout_prob"]
    loss = config["loss"]
    NN_md = md[config["model"]]
    is_logging_enabled = config["is_logging_enabled"]
    weighted_regression = config["weighted_regression"]
    yearlist = config["yearlist"]
    wandb_run_name = config["wandb_run_name"]
    preprocessed_path = config["preprocessed_data_path"]
    extra_input_features = config["extra_input_features"]
    station_regularization = config["station_regularization"]
    station_regularization_weight = config["station_regularization_weight"]
    imbalanced_regression_weight = config["imbalanced_regression_weight"]
    num_workers = config["num_workers"]
    num_devices = config["num_devices"]
    stn_reg_dmp = config["stn_reg_dmp"]

    if is_logging_enabled:
        wandb_logger = WandbLogger(
            project="geoeffectivenet", log_model=True, name=wandb_run_name
        )
    else:
        wandb_logger = False

    print("loading training data")
    if weighted_regression:
        train_ds = ShpericalHarmonicsDatasetPreprocessed(
            preprocessed_path,
            "train_with_weights",
            yearlist,
            weighted_regression,
            station_regularization,
        )
    else:
        train_ds = ShpericalHarmonicsDatasetPreprocessed(
            preprocessed_path,
            "train",
            yearlist,
            weighted_regression,
            station_regularization,
        )
    print("loading val data")
    val_ds = ShpericalHarmonicsDatasetPreprocessed(
        preprocessed_path, "val", yearlist, False, False
    )
    print("loading wiemer data")
    wiemer_ds = ShpericalHarmonicsDatasetPreprocessed(
        preprocessed_path, "wiemer", yearlist, False, False
    )

    print("load scalers")
    scaler = pickle.load(open(os.path.join(preprocessed_path, "scalers.p"), "rb"))

    wiemer_loader = data.DataLoader(
        wiemer_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("load supermag data")
    supermag_features = pickle.load(
        open(os.path.join(preprocessed_path, "supermag_features.p"), "rb")
    )

    print("load omni data")
    omni_features = pickle.load(
        open(os.path.join(preprocessed_path, "omni_features.p"), "rb")
    )

    targets_idx = [np.where(supermag_features == target)[0][0] for target in targets]

    # initialize model
    model = NN_md(
        past_omni_length,
        omni_features,
        supermag_features,
        omni_resolution,
        nmax,
        targets_idx,
        learning_rate=learning_rate,
        l2reg=l2reg,
        dropout_prob=dropout_prob,
        n_hidden=n_hidden,
        loss=loss,
        weighted_regression=weighted_regression,
        extra_input_features=extra_input_features,
        stn_reg=station_regularization,
        station_regularization_weight=station_regularization_weight,
        imbalanced_regression_weight=imbalanced_regression_weight,
        stn_reg_dmp=stn_reg_dmp,
    )
    model = model.double()

    # add wiemer data to the model to debug
    model.wiemer_data = wiemer_loader

    model.scaler = scaler

    checkpoint_path = wandb_run_name
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    pickle.dump(scaler, open(f"{checkpoint_path}/scalers.p", "wb"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_MSE", save_top_k=5
    )
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            devices=num_devices,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            max_epochs=max_epochs,
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="val_MSE", patience=100),
            ],
        )
    else:
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="val_MSE", patience=100),
            ],
        )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    experiment = Experiment(config_path)
    config = experiment.config
    print(f"Starting a run with {config}")
    train(config)
