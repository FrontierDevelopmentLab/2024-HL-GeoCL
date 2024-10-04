import os
import random
import time
from test import GeoCLoakDataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from model import DAGGERStationNet
from paths import (
    ace_mapping_train_files,
    dscovr_mapping_train_files,
    mapping_test_file_ace,
    mapping_test_file_dscovr,
    mapping_val_file_ace,
    mapping_val_file_dscovr,
    scaler_dir,
)
from torch.utils.data import ConcatDataset, DataLoader
from weighted_loss import WeightedMSELoss

BATCH_SIZE = 1024
NUM_WORKERS = 64


# Seed Initialization
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Loaders
ace_1 = GeoCLoakDataLoader(
    ace_mapping_train_files[0],
    "/home/chetrajpandey/daggerdata/ace_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)
ace_2 = GeoCLoakDataLoader(
    ace_mapping_train_files[1],
    "/home/chetrajpandey/daggerdata/ace_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ace_3 = GeoCLoakDataLoader(
    ace_mapping_train_files[2],
    "/home/chetrajpandey/daggerdata/ace_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

dscovr_1 = GeoCLoakDataLoader(
    dscovr_mapping_train_files[0],
    "/home/chetrajpandey/daggerdata/dscovr_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)
dscovr_2 = GeoCLoakDataLoader(
    dscovr_mapping_train_files[1],
    "/home/chetrajpandey/daggerdata/dscovr_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

dscovr_3 = GeoCLoakDataLoader(
    dscovr_mapping_train_files[2],
    "/home/chetrajpandey/daggerdata/dscovr_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ds_train = ConcatDataset([ace_1, dscovr_1])
ds_train_1 = ConcatDataset([ace_2, dscovr_2])
ds_train_2 = ConcatDataset([ace_3, dscovr_3])


ds_val_ace = GeoCLoakDataLoader(
    mapping_val_file_ace,
    "/home/chetrajpandey/daggerdata/ace_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ds_val_dscovr = GeoCLoakDataLoader(
    mapping_val_file_dscovr,
    "/home/chetrajpandey/daggerdata/dscovr_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ds_test_ace = GeoCLoakDataLoader(
    mapping_test_file_ace,
    "/home/chetrajpandey/daggerdata/ace_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ds_test_dscovr = GeoCLoakDataLoader(
    mapping_test_file_dscovr,
    "/home/chetrajpandey/daggerdata/dscovr_all.h5",
    "/home/chetrajpandey/daggerdata/",
    ["dbe", "dbn"],
    scaler_dir,
)

ds_val = ConcatDataset([ds_val_ace, ds_val_dscovr])
ds_test = ConcatDataset([ds_test_ace, ds_test_dscovr])

dataloader_train = DataLoader(
    ds_train,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=True,
)
dataloader_val = DataLoader(
    ds_val,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=False,
)
dataloader_test = DataLoader(
    ds_test,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=False,
)

cl_train_ds = [ds_train_1, ds_train_2]

model_params = {
    "input_dim": 29,
    "hidden_dim": 128,
    "output_dim": 1070,
    "fc_hidden_dim": 1024,
    "num_layers": 2,
    "dropout_prob": 0.3,
}

# Initialize Model
# device_ids = [0, 1, 2, 3]
model = DAGGERStationNet(**model_params).to(device)
# model = nn.DataParallel(net, device_ids=device_ids).to(device)

criterion = WeightedMSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.7, weight_decay=0.00001)


# EWC Implementation
class EWC:
    def __init__(self, model, dataloader, lambda_=0.7):
        self.model = model
        self.dataloader = dataloader
        self.lambda_ = lambda_
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.train()

        for inputs, targets, mask_target, kp_index in self.dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask_target = mask_target.to(device)
            kp_index = kp_index.to(device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = WeightedMSELoss()(
                outputs[mask_target], targets[mask_target], kp_index
            ).to(device)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n] += p.grad.data**2

        precision_matrices = {
            n: p / len(self.dataloader) for n, p in precision_matrices.items()
        }
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return self.lambda_ * loss


def compute_rmse(predictions, targets):
    # print(predictions.shape, targets.shape)
    return np.sqrt(((predictions - targets) ** 2).mean())


def evaluate_and_log(model, data_loader, criterion, stage, dataset_name):
    model.eval()
    total_loss = 0
    avg_loss, avg_rmse = 0, 0
    predictions_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets, mask_target, kp_index in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask_target = mask_target.to(device)
            kp_index = kp_index.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[mask_target], targets[mask_target], kp_index).to(
                device
            )
            total_loss += loss.item()
            predictions_list.extend(outputs[mask_target].cpu().detach().numpy())
            targets_list.extend(targets[mask_target].cpu().detach().numpy())

    avg_loss = total_loss / len(data_loader)
    avg_rmse = compute_rmse(np.array(predictions_list), np.array(targets_list))
    print(f"{dataset_name} Loss: {avg_loss}, {dataset_name} RMSE: {avg_rmse}")

    return avg_loss, avg_rmse


def train_and_evaluate(
    model, ewc, train_loader, val_loader, test_loader, epochs=25, stage=1
):

    # Initialize tracking variables for best model
    best_loss = float("inf")
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        val_loss, val_rmse, test_loss, test_rmse, avg_train_loss, avg_train_loss = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        predictions_list = []
        targets_list = []

        for inputs, targets, mask_target, kp_index in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)
            mask_target = mask_target.to(device)
            kp_index = kp_index.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[mask_target], targets[mask_target], kp_index).to(
                device
            )
            if ewc:
                loss += ewc.penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions_list.extend(outputs[mask_target].cpu().detach().numpy())
            targets_list.extend(targets[mask_target].cpu().detach().numpy())
            # print(total_loss)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_rmse = compute_rmse(
            np.array(predictions_list), np.array(targets_list)
        )

        # Evaluate
        val_loss, val_rmse = evaluate_and_log(
            model, val_loader, criterion, stage, "Validation"
        )
        test_loss, test_rmse = evaluate_and_log(
            model, test_loader, criterion, stage, "Test"
        )

        # Update best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Log metrics
        global_metrics["train_loss"].append(avg_train_loss)
        global_metrics["train_rmse"].append(avg_train_rmse)
        global_metrics["val_loss"].append(val_loss)
        global_metrics["val_rmse"].append(val_rmse)
        global_metrics["test_loss"].append(test_loss)
        global_metrics["test_rmse"].append(test_rmse)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}, Train RMSE: {avg_train_rmse}")
        print(f"Validation Loss: {val_loss}, Validation RMSE: {val_rmse}")
        print(f"Test Loss: {test_loss}, Test RMSE: {test_rmse}")

    print("BEST VAL LOSS: ", best_loss)

    return best_loss, best_model_state


def save_model_and_fisher(model, ewc, stage):
    os.makedirs("models_new", exist_ok=True)
    torch.save(model.state_dict(), f"models_new/model_stage_{stage}.pth")
    fisher_dict = {n: p.cpu().numpy() for n, p in ewc._precision_matrices.items()}
    means_dict = {n: p.cpu().numpy() for n, p in ewc._means.items()}
    torch.save(
        {"fisher": fisher_dict, "means": means_dict},
        f"models_new/fisher_stage_{stage}.pth",
    )


def load_model_and_fisher(model, stage):
    model.load_state_dict(torch.load(f"models_new/model_stage_{stage}.pth"))
    fisher_info = torch.load(f"models_new/fisher_stage_{stage}.pth")
    fisher_dict = {
        n: torch.tensor(p).to(model.parameters().__next__().device)
        for n, p in fisher_info["fisher"].items()
    }
    means_dict = {
        n: torch.tensor(p).to(model.parameters().__next__().device)
        for n, p in fisher_info["means"].items()
    }
    return fisher_dict, means_dict


# Metrics Storage
global_metrics = {
    "train_loss": [],
    "train_rmse": [],
    "val_loss": [],
    "val_rmse": [],
    "test_loss": [],
    "test_rmse": [],
}

# Continual Learning Stages
# Initial Training
start_time = time.time()
ewc = None
best_loss, best_model_state = train_and_evaluate(
    model, ewc, dataloader_train, dataloader_val, dataloader_test
)

# Save initial model and Fisher Information
model.load_state_dict(best_model_state)
ewc = EWC(model, dataloader_train)
save_model_and_fisher(model, ewc, stage=1)
end_time = time.time()


# For additional training stages with new data
for stage, ds in enumerate(cl_train_ds):  # Assuming 3 stages
    print(f"Training Stage {stage+2}")

    # Load new train data
    train_loader_new = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    # Load model and Fisher Information from previous stage
    fisher_dict, means_dict = load_model_and_fisher(model, stage + 1)

    # Update EWC with new data
    ewc = EWC(model, train_loader_new)
    ewc._precision_matrices = fisher_dict
    ewc._means = means_dict

    # Train with new data including EWC penalty
    best_loss, best_model_state = train_and_evaluate(
        model, ewc, train_loader_new, dataloader_val, dataloader_test
    )

    # Save model and Fisher Information for the new stage
    model.load_state_dict(best_model_state)
    save_model_and_fisher(model, ewc, stage + 2)


def save_best_metrics_as_csv(
    best_metrics, filename="models_new/best_metrics_trial.csv"
):
    df = pd.DataFrame(best_metrics)
    df.to_csv(filename, index=False)
    print(f"Best metrics saved to {filename}")


# Call the function to save the best metrics
save_best_metrics_as_csv(global_metrics)


# Plot all collected metrics at the end
def plot_bar_metrics():
    # Define stages to plot
    stages_indices = [100, 200, 300]
    stages_labels = ["Stage 1", "Stage 2", "Stage 3"]

    # Extract metrics for specified stages
    train_loss = [global_metrics["train_loss"][i - 1] for i in stages_indices]
    val_loss = [global_metrics["val_loss"][i - 1] for i in stages_indices]
    test_loss = [global_metrics["test_loss"][i - 1] for i in stages_indices]
    train_rmse = [global_metrics["train_rmse"][i - 1] for i in stages_indices]
    val_rmse = [global_metrics["val_rmse"][i - 1] for i in stages_indices]
    test_rmse = [global_metrics["test_rmse"][i - 1] for i in stages_indices]

    # Define the bar width and positions
    bar_width = 0.15
    index = np.arange(len(stages_labels))

    # Plot
    plt.figure(figsize=(16, 8))

    # Bars for losses
    plt.bar(
        index - 1.5 * bar_width, train_loss, bar_width, label="Train Loss", color="b"
    )
    plt.bar(
        index - 0.5 * bar_width, val_loss, bar_width, label="Validation Loss", color="g"
    )
    plt.bar(index + 0.5 * bar_width, test_loss, bar_width, label="Test Loss", color="r")

    # Bars for RMSE
    plt.bar(
        index + 1.5 * bar_width, train_rmse, bar_width, label="Train RMSE", color="c"
    )
    plt.bar(
        index + 2.5 * bar_width, val_rmse, bar_width, label="Validation RMSE", color="m"
    )
    plt.bar(index + 3.5 * bar_width, test_rmse, bar_width, label="Test RMSE", color="y")

    # Labeling
    plt.xlabel("Stage")
    plt.ylabel("Loss/RMSE")
    plt.title("Metrics at Specific Stages")
    plt.xticks(index + 2 * bar_width, stages_labels)
    plt.legend()

    # Save and show plot
    plt.savefig("models_new/metrics_barplot_trial.png")
    plt.show()


plot_bar_metrics()
