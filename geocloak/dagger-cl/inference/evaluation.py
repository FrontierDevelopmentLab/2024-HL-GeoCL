import torch
from dataloader_eval import GeoCLoakDataLoader
from model import DAGGERStationNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

geo_data_loader = GeoCLoakDataLoader(
    mapping_file="/home/jupyter/ace_processed/ace_mapping_val.csv",
    input_path="/home/jupyter/ace_processed/ace_processed",
    target_paths=[
        "/home/jupyter/supermag_processed/dbe_geo",
        "/home/jupyter/supermag_processed/dbn_geo",
    ],
    scaler_dir="/home/jupyter/models/DAGGER_CL",
)

batch_size = 1  # Example batch size
data_loader = DataLoader(
    geo_data_loader,
    batch_size=batch_size,
    num_workers=80,
    pin_memory=True,
    shuffle=False,
)

model = DAGGERStationNet(
    input_dim=29, hidden_dim=128, output_dim=1070, fc_hidden_dim=1024, num_layers=1
).to(device)
model.load_state_dict(torch.load("../models/model_stage_1.pth"))
model.eval()

total_rmse = 0
total_mae = 0
total_r2 = 0
num_batches = 0

with torch.no_grad():
    for inputs, targets, mask in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        outputs = model(inputs)

        outputs = outputs[mask].detach().cpu().numpy()
        targets = targets[mask].detach().cpu().numpy()

        # Calculate RMSE, MAE, and R² for this batch
        batch_rmse = mean_squared_error(targets, outputs, squared=False)
        batch_mae = mean_absolute_error(targets, outputs)
        batch_r2 = r2_score(targets, outputs)

        total_rmse += batch_rmse
        total_mae += batch_mae
        total_r2 += batch_r2
        num_batches += 1

# Calculate the mean for the entire dataset
mean_rmse = total_rmse / num_batches
mean_mae = total_mae / num_batches
mean_r2 = total_r2 / num_batches

print(f"Mean RMSE: {mean_rmse}")
print(f"Mean MAE: {mean_mae}")
print(f"Mean R²: {mean_r2}")
