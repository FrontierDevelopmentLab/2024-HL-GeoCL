import os
import torch
import numpy as np
import pandas as pd
from model import DAGGERStationNet
from dataloader_eval import GeoCLoakDataLoader
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BATCH_SIZE = 1024
NUM_WORKERS = 64

# Define file paths for the data and model checkpoints
ace_mapping_train_files = [
    '/home/chetrajpandey/data/ace_mapping_sample_1.csv',
    '/home/chetrajpandey/data/ace_mapping_sample_2.csv',
    '/home/chetrajpandey/data/ace_mapping_sample_3.csv',
]
dscovr_mapping_train_files = [
    '/home/chetrajpandey/data/dscovr_mapping_sample_1.csv',
    '/home/chetrajpandey/data/dscovr_mapping_sample_2.csv',
    '/home/chetrajpandey/data/dscovr_mapping_sample_3.csv',
]
mapping_val_file_dscovr = '/home/jupyter/dscovr_processed/dscovr_mapping_val.csv'
mapping_test_file_dscovr = '/home/jupyter/dscovr_processed/dscovr_mapping_test.csv'

mapping_val_file_ace = '/home/jupyter/ace_processed/ace_mapping_val.csv'
mapping_test_file_ace = '/home/jupyter/ace_processed/ace_mapping_test.csv'

scaler_dir = "/home/jupyter/models/DAGGER_CL"

# Data Loaders
ace_1 = GeoCLoakDataLoader(ace_mapping_train_files[0], '/home/chetrajpandey/daggerdata/ace_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)
ace_2 = GeoCLoakDataLoader(ace_mapping_train_files[1], '/home/chetrajpandey/daggerdata/ace_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ace_3 = GeoCLoakDataLoader(ace_mapping_train_files[2], '/home/chetrajpandey/daggerdata/ace_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

dscovr_1 = GeoCLoakDataLoader(dscovr_mapping_train_files[0], '/home/chetrajpandey/daggerdata/dscovr_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)
dscovr_2 = GeoCLoakDataLoader(dscovr_mapping_train_files[1], '/home/chetrajpandey/daggerdata/dscovr_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

dscovr_3 = GeoCLoakDataLoader(dscovr_mapping_train_files[2], '/home/chetrajpandey/daggerdata/dscovr_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ds_train_1 = ConcatDataset([ace_1, dscovr_1])
ds_train_2 = ConcatDataset([ace_2, dscovr_2])
ds_train_3 = ConcatDataset([ace_3, dscovr_3])

ds_val_ace = GeoCLoakDataLoader(mapping_val_file_ace, '/home/chetrajpandey/daggerdata/ace_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ds_val_dscovr = GeoCLoakDataLoader(mapping_val_file_dscovr, '/home/chetrajpandey/daggerdata/dscovr_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ds_test_ace = GeoCLoakDataLoader(mapping_test_file_ace, '/home/chetrajpandey/daggerdata/ace_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ds_test_dscovr = GeoCLoakDataLoader(mapping_test_file_dscovr, '/home/chetrajpandey/daggerdata/dscovr_all.h5',
                 '/home/chetrajpandey/daggerdata/', ['dbe', 'dbn'], scaler_dir)

ds_val = ConcatDataset([ds_val_ace, ds_val_dscovr])
ds_test = ConcatDataset([ds_test_ace, ds_test_dscovr])

dataloader_train_1 = DataLoader(ds_train_1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)
dataloader_train_2 = DataLoader(ds_train_2, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)
dataloader_train_3 = DataLoader(ds_train_3, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)

dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)
dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to evaluate the model
def evaluate_model(model, dataloader):
    total_mse, total_mae, total_r2 = 0, 0, 0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets, mask in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs[mask].cpu().numpy()
            targets = targets[mask].cpu().numpy()

            mse = mean_squared_error(targets, outputs)
            mae = mean_absolute_error(targets, outputs)
            r2 = r2_score(targets, outputs)

            total_mse += mse
            total_mae += mae
            total_r2 += r2
            num_batches += 1

    # Calculate RMSE
    avg_mse = total_mse / num_batches
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = total_mae / num_batches
    avg_r2 = total_r2 / num_batches

    return avg_mse, avg_rmse, avg_mae, avg_r2

# Evaluate and save results for each stage
stages = {
    'stage_1': {'model_checkpoint': '../models/model_stage_1.pth', 'train_loader': dataloader_train_1},
    'stage_2': {'model_checkpoint': '../models/model_stage_2.pth', 'train_loader': dataloader_train_2},
    'stage_3': {'model_checkpoint': '../models/model_stage_3.pth', 'train_loader': dataloader_train_3},
}

results = []
for stage, data in stages.items():

    #
    model = DAGGERStationNet(input_dim=29, hidden_dim=128, output_dim=1070, fc_hidden_dim=1024, num_layers=1).to(device)
    # Load model checkpoint
    model.load_state_dict(torch.load(data['model_checkpoint']))
    
    # Evaluate on train, validation, and test sets
    mse_train, rmse_train, mae_train, r2_train = evaluate_model(model, data['train_loader'])
    mse_val, rmse_val, mae_val, r2_val = evaluate_model(model, dataloader_val)
    mse_test, rmse_test, mae_test, r2_test = evaluate_model(model, dataloader_test)

    # Collect results
    results.append({
        'Stage': stage,
        'MSE_Train': mse_train, 'RMSE_Train': rmse_train, 'MAE_Train': mae_train, 'R2_Train': r2_train,
        'MSE_Val': mse_val, 'RMSE_Val': rmse_val, 'MAE_Val': mae_val, 'R2_Val': r2_val,
        'MSE_Test': mse_test, 'RMSE_Test': rmse_test, 'MAE_Test': mae_test, 'R2_Test': r2_test,
    })

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_metrics.csv', index=False)
