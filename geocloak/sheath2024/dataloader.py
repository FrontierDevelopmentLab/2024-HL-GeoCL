import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class SHEATHDataLoader(Dataset):
    def __init__(self, directory, filename):
        # Read the CSV data from the local directory
        file_path = os.path.join(directory, filename)
        self.dataframe = pd.read_csv(file_path)

        # Select input features and target variables
        self.inputs = self.dataframe.iloc[
            :, 2:-7
        ].values  # Assuming columns 2 to the one before the last 7 are inputs
        self.targets = self.dataframe.iloc[
            :, -7:
        ].values  # Assuming the last 7 columns are targets

        # Check for NaN or infinite values and handle them if necessary
        assert not np.any(np.isnan(self.inputs)), "Inputs contain NaN values"
        assert not np.any(np.isinf(self.inputs)), "Inputs contain infinite values"
        assert not np.any(np.isnan(self.targets)), "Targets contain NaN values"
        assert not np.any(np.isinf(self.targets)), "Targets contain infinite values"

        # Initialize scalers
        self.scaler_inputs = StandardScaler()
        self.scaler_targets = StandardScaler()

        # Fit and transform inputs and targets
        self.inputs = self.scaler_inputs.fit_transform(self.inputs)
        self.targets = self.scaler_targets.fit_transform(self.targets)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_data, target_data




""" USE CASE"""
# def main():
#     # Specifying the local directory
#     directory = "/home/chetrajpandey/data/formatted_data/sheath_splits"

#     # Create datasets
#     train_dataset = SHEATHDataLoader(directory, 'train_set.csv')
#     val_dataset = SHEATHDataLoader(directory, 'val_set.csv')
#     test_dataset = SHEATHDataLoader(directory, 'test_set.csv')

#     # Print the number of samples in each dataset
#     print(f"Training set length: {len(train_dataset)}")
#     print(f"Validation set length: {len(val_dataset)}")
#     print(f"Test set length: {len(test_dataset)}")

#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = main()
