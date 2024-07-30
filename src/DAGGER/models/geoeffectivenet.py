import numpy as np
import torch
import torch.nn as nn
from models.base import BaseModel
from models.spherical_harmonics import SphericalHarmonics


class NamedAccess:
    def __init__(self, data, columns):
        self.columns = columns
        self.data = data

    def __getitem__(self, key):
        key = np.where(self.columns == key)[0][0]
        return self.data[..., key]


class NeuralRNNWiemer(BaseModel):
    def __init__(
        self,
        past_omni_length,
        omni_features,
        supermag_features,
        omni_resolution,
        nmax,
        targets_idx,
        extra_input_features,
        **kwargs
    ):
        super(NeuralRNNWiemer, self).__init__(**kwargs)

        # idx of targets in dataset
        self.targets_idx = targets_idx

        self.extra_input_features = extra_input_features

        self.omni_resolution = omni_resolution

        hidden = kwargs.pop("n_hidden", 8)
        dropout_prob = kwargs.pop("dropout", 0.5)
        levels = 2
        kernel_size = 24
        levels = levels
        kernel_size = kernel_size
        [hidden] * levels

        self.omni_past_encoder = nn.GRU(
            input_size=25 + len(extra_input_features),
            hidden_size=hidden,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )


        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, n_coeffs * len(targets_idx), bias=False),  # needs to map to SuperMAG
        )

        self.omni_features = omni_features
        self.supermag_features = supermag_features

    def forward(
        self, past_omni, past_supermag, mlt, mcolat, dates, future_dates, **kargs
    ):
        # 10 mins average
        past_omni = nn.AvgPool1d(self.omni_resolution)(
            past_omni.permute([0, 2, 1])
        ).permute([0, 2, 1])

        past_omni = NamedAccess(past_omni, self.omni_features)

        features = past_omni # get this from the .csv file

        # zero fill
        features[features.isnan()] = 0.0

        assert not (torch.isnan(features).any() or torch.isinf(features).any())

        encoded = self.omni_past_encoder(features)[1][0]

        predictions = self.encoder_mlp(encoded).reshape(
            encoded.shape[0], -1, len(self.targets_idx)
        )

        return predictions
