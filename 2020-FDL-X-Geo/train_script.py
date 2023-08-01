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

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

hyperparameter_defaults = dict(future_length = 1, past_omni_length = 120,
                                omni_resolution = 1, nmax = 20,lag = 30,
                                learning_rate = 5e-03,batch_size = 2500,
                                l2reg=1e-3,epochs = 100, dropout_prob=0.3,n_hidden=8,
                                loss='MAE',model='NeuralRNNWiemer',stn_reg=True,
                                is_logging_enabled = True,
                                extra_input_features = ["SME", "SML", "SMU", "SMR"],  # Make sure this is a subset of what you had in preprocess.py
                          )
                                # learning_rate originally 1e-5
  
md = {'NeuralRNNWiemer_HidddenSuperMAG':NeuralRNNWiemer_HidddenSuperMAG,
        'NeuralRNNWiemer':NeuralRNNWiemer}

yearlist = np.arange(2013, 2014+1)

preprocessed_path = './processed_data_all_years'

#----- Data loading also depends on the sweep parameters.
#----- Hence this process will be repeated per training cycle.
def train(config):
    print(config)
    future_length = config["future_length"]
    past_omni_length = config["past_omni_length"]
    omni_resolution = config["omni_resolution"]
    nmax = config["nmax"]
    targets = ["dbe_nez", "dbn_nez"] #config.targets
    lag = config["lag"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    l2reg=config["l2reg"]
    max_epochs = config["epochs"]
    n_hidden=config["n_hidden"]
    dropout_prob=config["dropout_prob"]
    loss = config["loss"]
    NN_md = md[config["model"]]
    is_logging_enabled = config["is_logging_enabled"]
    stn_reg = config["stn_reg"] 
    extra_input_features = config["extra_input_features"]

    wandb_run_name = f"RegularizationTest_20132014_{config['model']}_{loss}_{past_omni_length}_{nmax}_{n_hidden}_{learning_rate*1e6}_{l2reg*1e6}"
    if is_logging_enabled:
        wandb_logger = WandbLogger(project="geoeffectivenet", log_model=True, name=wandb_run_name)
    else:
        wandb_logger = None
        
    train_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'train', yearlist)
    val_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'val', yearlist)
    wiemer_ds = ShpericalHarmonicsDatasetPreprocessed(preprocessed_path, 'wiemer', yearlist)
    
    scaler = pickle.load(open(os.path.join(preprocessed_path, 'scalers.p'), 'rb'))

    wiemer_loader = data.DataLoader(
        wiemer_ds, batch_size=batch_size, shuffle=False, num_workers=32
    )
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=32
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=32
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
        targets_idx,
        extra_input_features,
        learning_rate = learning_rate,
        l2reg=l2reg,
        dropout_prob=dropout_prob,
        n_hidden=n_hidden,
        loss=loss,
        stn_reg=stn_reg
    )
    model = model.double()

    # add wiemer data to the model to debug
    model.wiemer_data = wiemer_loader
    # model.test_data = test_loader
    model.scaler = scaler

    # save the scaler to de-standarize prediction
    # checkpoint_path = f"checkpoints_{int(learning_rate*1e5)}_{int(batch_size)}_{int(l2reg*1e6)}_{nmax}_{loss}"
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
    
    config = hyperparameter_defaults
    print(f'Starting a run with {config}')
    train(config)
    