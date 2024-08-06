import torch
import torch.nn as nn
import torch.nn.init as init

class SHEATH_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3, init_type='kaiming'):
        super(SHEATH_MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Apply the chosen initialization method
        self._initialize_weights(init_type)
    
    def _initialize_weights(self, init_type):
        if init_type == 'kaiming':
            init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        elif init_type == 'xavier':
            init.xavier_normal_(self.fc1.weight)
            init.xavier_normal_(self.fc2.weight)
            init.xavier_normal_(self.fc3.weight)
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
