import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class DAGGERStationNet(nn.Module):
    """
    DAGGERStationNet: A neural network model combining GRU and fully connected layers.

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input.
    hidden_dim : int
        The number of features in the hidden state of the GRU.
    output_dim : int
        The number of output features.
    fc_hidden_dim : int, optional
        The number of features in the hidden layer of the fully connected block (default is 512).
    num_layers : int, optional
        The number of recurrent layers in the GRU (default is 2).
    dropout_prob : float, optional
        The dropout probability (default is 0.3).

    Attributes
    ----------
    gru : torch.nn.GRU
        The GRU layer.
    fc : torch.nn.Sequential
        The fully connected block with an ELU activation and dropout layer.
    hidden_dim : int
        The number of features in the hidden state of the GRU.
    num_layers : int
        The number of recurrent layers in the GRU.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        fc_hidden_dim=512,
        num_layers=2,
        dropout_prob=0.3,
    ):
        super(DAGGERStationNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Sequential block with additional layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),  # Configurable hidden dimension
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(fc_hidden_dim, output_dim),  # Final output layer
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device
        )  # Initial hidden state
        out, _ = self.gru(x, h0)  # GRU forward pass
        out = self.fc(
            out[:, -1, :]
        )  # Take the last time step output and pass it through the Sequential block
        return out
