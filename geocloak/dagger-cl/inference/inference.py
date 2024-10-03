import os
import torch
import numpy as np
from model import DAGGERStationNet
from dataloader_inference import GeoCLoakDataLoader
from torch.utils.data import DataLoader
from unscale_targets import unscale_predicted_target

parent_dir = '/home/inference_outputs/DAGGER/original_testset/'

def save_predictions(model_checkpoint_index, filename, dbe_data=None, dbn_data=None):
    # Create main directory based on model_checkpoint_index
    stage_dir = os.path.join(parent_dir, f'stage_{model_checkpoint_index}')
    if not os.path.exists(stage_dir):
        os.makedirs(stage_dir)
    
    # Create ace_predictions directory
    ace_predictions_dir = os.path.join(stage_dir, 'ace_predictions')
    if not os.path.exists(ace_predictions_dir):
        os.makedirs(ace_predictions_dir)
    
    # Create dbe_geo or dbn_geo directories and save CSVs based on available data
    if dbe_data is not None:
        dbe_geo_dir = os.path.join(ace_predictions_dir, 'dbe_geo')
        if not os.path.exists(dbe_geo_dir):
            os.makedirs(dbe_geo_dir)
        # Save dbe_data to CSV with filename
        dbe_file_path = os.path.join(dbe_geo_dir, filename)
        np.savetxt(dbe_file_path, dbe_data, delimiter=',', fmt='%.6f')
    
    if dbn_data is not None:
        dbn_geo_dir = os.path.join(ace_predictions_dir, 'dbn_geo')
        if not os.path.exists(dbn_geo_dir):
            os.makedirs(dbn_geo_dir)
        # Save dbn_data to CSV with filename
        dbn_file_path = os.path.join(dbn_geo_dir, filename)
        np.savetxt(dbn_file_path, dbn_data, delimiter=',', fmt='%.6f')

def main(model_checkpoint_index):
   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geo_data_loader = GeoCLoakDataLoader(
        mapping_file='/home/jupyter/ace_processed/ace_mapping_test.csv',
        # mapping_file='/home/chetrajpandey/data/test_samples/dscovr_test_sample.csv',
        input_path='/home/jupyter/ace_processed/ace_processed',
        scaler_dir="/home/jupyter/models/DAGGER_CL",
    )

    batch_size = 1
    data_loader = DataLoader(geo_data_loader, batch_size=batch_size, num_workers=80, pin_memory=True, shuffle=False)

    model = DAGGERStationNet(input_dim=29, hidden_dim=128, output_dim=1070, fc_hidden_dim=1024, num_layers=1).to(device)
    model.load_state_dict(torch.load(f'../models/model_stage_{model_checkpoint_index}.pth'))
    model.eval()

    with torch.no_grad():
        for inputs, filename in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            filename = filename[0]
            print(filename)
            outputs = outputs.squeeze(0).detach().cpu()
            dbe, dbn = unscale_predicted_target(outputs)
            
            # Save predictions with filename
            save_predictions(model_checkpoint_index, filename, dbe, dbn)

if __name__ == '__main__':
    # Example usage, replace with actual model checkpoint index
    model_checkpoint_index = 3
    main(model_checkpoint_index)
