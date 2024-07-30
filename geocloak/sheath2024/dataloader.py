import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from gcp_read_write_utils import read_csv_from_gcs
import numpy as np

class SHEATHDataLoader(Dataset):
    def __init__(self, bucket, folder, filename):
        # Read the CSV data from GCS
        self.dataframe = read_csv_from_gcs(bucket_name=bucket, source_folder=folder, file_name=filename)
        
        # Select input features and target variables
        self.inputs = self.dataframe.iloc[:, 2:-7].values  # Assuming columns 2 to the one before the last 7 are inputs
        self.targets = self.dataframe.iloc[:, -7:].values  # Assuming the last 7 columns are targets
        
        # Check for NaN or infinite values and handle them if necessary
        assert not np.any(np.isnan(self.inputs)), "Inputs contain NaN values"
        assert not np.any(np.isinf(self.inputs)), "Inputs contain infinite values"
        assert not np.any(np.isnan(self.targets)), "Targets contain NaN values"
        assert not np.any(np.isinf(self.targets)), "Targets contain infinite values"
        
        # Print raw data statistics
        # print("Raw inputs mean:", self.inputs.mean(axis=0))
        # print("Raw inputs std:", self.inputs.std(axis=0))
        # print("Raw targets mean:", self.targets.mean(axis=0))
        # print("Raw targets std:", self.targets.std(axis=0))
        
        # Initialize scalers
        self.scaler_inputs = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        # Fit and transform inputs and targets
        self.inputs = self.scaler_inputs.fit_transform(self.inputs)
        self.targets = self.scaler_targets.fit_transform(self.targets)
        
        # Print scaled data statistics
        # print("Scaled inputs mean:", self.inputs.mean(axis=0))
        # print("Scaled inputs std:", self.inputs.std(axis=0))
        # print("Scaled targets mean:", self.targets.mean(axis=0))
        # print("Scaled targets std:", self.targets.std(axis=0))
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_data, target_data

# # Create data loaders
# bucket = 'geocloak2024'
# train_dataset = SHEATHDataLoader(bucket, 'formatted_data/sheath_splits', 'train_set.csv')
# val_dataset = SHEATHDataLoader(bucket, 'formatted_data/sheath_splits', 'val_set.csv')
# test_dataset = SHEATHDataLoader(bucket, 'formatted_data/sheath_splits', 'test_set.csv')
# print(len(train_dataset))

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

