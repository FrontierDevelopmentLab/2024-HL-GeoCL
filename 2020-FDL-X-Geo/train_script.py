import os.path
import pickle
import pandas as pd
import h5py
import numpy as np
import pytorch_lightning as pl
import torch.optim
import wandb
from astropy.time import Time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.geoeffectivenet import *
from models.spherical_harmonics import SphericalHarmonics
from utils.data_utils import get_iaga_data, load_cached_data,get_wiemer_data,get_iaga_data_as_list
from utils.splitter import generate_indices
from dataloader import ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset, ShpericalHarmonicsDatasetPreprocessed
from experiment import Experiment
torch.set_default_dtype(torch.float64)  # this is important else it will overflow

  
md = {'NeuralRNNWiemer_HidddenSuperMAG':NeuralRNNWiemer_HidddenSuperMAG,
        'NeuralRNNWiemer':NeuralRNNWiemer}


config_path = 'experiment.yaml'

#----- Data loading also depends on the sweep parameters.
#----- Hence this process will be repeated per training cycle.
def train(config):
    future_length = config["future_length"]
    past_omni_length = config["past_omni_length"]
    omni_resolution = config["omni_resolution"]
    nmax = config["nmax"]
    targets = ["dbe_nez", "dbn_nez"] #config.targets
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    l2reg=config["l2reg"]
    max_epochs = config["epochs"]
    n_hidden=config["n_hidden"]
    dropout_prob=config["dropout_prob"]
    loss = config["loss"]
    NN_md = md[config["model"]]
    is_logging_enabled = config["is_logging_enabled"]
    weighted_regression = config["weighted_regression"]
    yearlist = config["yearlist"]  
    wandb_run_name = config["wandb_run_name"]
    preprocessed_path = config["preprocessed_data_path"]
    extra_input_features = config["extra_input_features"]
    station_regularization = config["station_regularization"]

    wandb_run_name = f"Imbalanced_Regression_2013_2014_new_scale"
    if is_logging_enabled:
        wandb_logger = WandbLogger(project="geoeffectivenet", log_model=True, name=wandb_run_name)
    else:
        wandb_logger = False
    
    if weighted_regression:
        train_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'train_with_weights', yearlist, station_regularization)
    else:
        train_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'train', yearlist, station_regularization)
    val_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'val', yearlist, station_regularization)
    wiemer_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'wiemer', yearlist, False)

    wiemer_loader = data.DataLoader(
        wiemer_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    plot_loader = data.DataLoader(val_ds, batch_size=4, shuffle=False)
    supermag_features = pickle.load(open(os.path.join(preprocessed_path, 'supermag_features.p'), 'rb'))
    omni_features = pickle.load(open(os.path.join(preprocessed_path, 'omni_features.p'), 'rb'))
    
    targets_idx = [np.where(supermag_features == target)[0][0] for target in targets]

    # initialize model
    model = NN_md(
        past_omni_length,
        future_length,
        omni_features,
        supermag_features,
        omni_resolution,
        nmax,
        targets_idx,learning_rate = learning_rate,
        l2reg=l2reg,
        dropout_prob=dropout_prob,
        n_hidden=n_hidden,
        loss=loss,
        weighted_regression=weighted_regression,
        extra_input_features=extra_input_features,
        stn_reg=station_regularization
    )
    model = model.double()

    # add wiemer data to the model to debug
    model.wiemer_data = wiemer_loader
    
    model.scaler = scaler

    checkpoint_path = wandb_run_name
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    pickle.dump(scaler, open(f'{checkpoint_path}/scalers.p', "wb"))
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="val_MSE", save_top_k=5)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
        devices=2,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='val_MSE',patience = 100)]
    )
    else:
        trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='val_MSE',patience = 100)]
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    experiment = Experiment(config_path)
    config = experiment.config
    print(f'Starting a run with {config}')
    train(config)
