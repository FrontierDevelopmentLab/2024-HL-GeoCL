import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from generate_data import generate_sine_wave

# Define the LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(data, labels):
    model = LSTMPredictor(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_data = TensorDataset(torch.tensor(data).float(), torch.tensor(labels).float())
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    for epoch in range(100):
        for seq, targets in train_loader:
            optimizer.zero_grad()
            output = model(seq.unsqueeze(-1))
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    times, data = generate_sine_wave(freq=1, sample_rate=100, duration=10)
    # Preparing data (assuming sliding window of size 10)
    inputs = np.array([data[i:i+10] for i in range(len(data)-10)])
    targets = np.array(data[10:])
    model = train_model(inputs, targets)
    torch.save(model.state_dict(), 'lstm_model.pth')
