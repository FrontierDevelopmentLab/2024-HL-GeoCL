import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Step 1: Load the data
data = pd.read_csv('evaluation_metrics.csv')

# Apply dark background
plt.style.use('dark_background')

# Step 2: Calculate standard deviations for each metric across the stages
metrics = ['MSE', 'RMSE', 'MAE']
std_devs = {}

for metric in metrics:
    std_devs[f'{metric}_Train_SD'] = data[[f'{metric}_Train']].std().values[0]
    std_devs[f'{metric}_Val_SD'] = data[[f'{metric}_Val']].std().values[0]

# Step 3: Set up the plot with two rows: one for the metrics, one for the standard deviations
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns layout

# Custom y-axis limits for metrics
y_limits = {
    'MSE': (0, 1.5),
    'RMSE': (0, 1.5),
    'MAE': (0, 1)
}

colors = ['b', 'g', 'r']
stages = data['Stage']

# Step 4: Plot the metrics as bar charts in the first row
for i, metric in enumerate(metrics):
    ax = axs[0, i]
    
    # Plot bars for Train and Val metrics
    bar_width = 0.35
    indices = np.arange(len(stages))
    
    ax.bar(indices - bar_width/2, data[f'{metric}_Train'], bar_width, label=f'{metric} Train', color=colors[i], alpha=0.6)
    ax.bar(indices + bar_width/2, data[f'{metric}_Val'], bar_width, label=f'{metric} Val', color=colors[i], alpha=0.6, hatch='//')
    
    # Set y-axis limits
    ax.set_ylim(y_limits[metric])
    
    # Control y-tick intervals within the limits
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Set to a maximum of 5 ticks within the limits
    
    ax.set_title(f'{metric} Over CL Stages')
    ax.set_xlabel('CL Stages')
    ax.set_ylabel(metric)
    ax.set_xticks(indices)
    ax.set_xticklabels(stages, rotation=0)
    ax.legend()
    ax.grid(True, color='gray', linestyle='--', axis='y')  # Set grid color and only display horizontal grid lines

# Step 5: Plot the standard deviations in the second row
for i, metric in enumerate(metrics):
    ax = axs[1, i]
    
    # Plot standard deviation as a horizontal bar across stages
    train_sd = std_devs[f'{metric}_Train_SD']
    val_sd = std_devs[f'{metric}_Val_SD']
    
    ax.bar('Train', train_sd, color=colors[i], alpha=0.6, label=f'{metric} Train SD')
    ax.bar('Val', val_sd, color=colors[i], alpha=0.6, hatch='//', label=f'{metric} Val SD')
    
    ax.set_ylim(0, max(train_sd, val_sd) * 1.5)  # Set y-limit based on SD
    ax.set_title(f'{metric} Variability Across CL Stages')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.grid(True, color='gray', linestyle='--', axis='y')

# Adjust layout for readability
plt.tight_layout()
plt.savefig('metrics_with_separate_sd.jpg', dpi=300)
plt.show()
