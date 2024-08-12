import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import LSTMPredictor, generate_sine_wave
import pandas as pd
import subprocess
import datetime
from google.cloud import storage


def get_current_datetime():
    # Returns a formatted datetime string, e.g., '20230901_150505'
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

'''''
def predict_and_plot(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data).float().unsqueeze(-1))
    return predictions.numpy().flatten()


if __name__ == "__main__":
    times, data = generate_sine_wave(freq=1, sample_rate=100, duration=10)
    inputs = np.array([data[i:i + 10] for i in range(len(data) - 10)])
    model = LSTMPredictor(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    model.load_state_dict(torch.load('lstm_model.pth'))

    predictions = predict_and_plot(model, inputs)

    plt.figure(figsize=(10, 5))
    plt.plot(times[10:], data[10:], label='Actual')
    plt.plot(times[10:], predictions, label='Predicted', linestyle='--')
    plt.title("Time Series Prediction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
'''''


def upload_to_gcloud(project_name, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def predict_and_plot(model, data, inputs_save_path, predictions_save_path):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data).float().unsqueeze(-1))
    predictions = predictions.numpy().flatten()

    # Save predictions to CSV
    df = pd.DataFrame({'Actual': data[:, -1], 'Predicted': predictions})
    df.to_csv(predictions_save_path, index=False)
    upload_to_gcloud('hl-geo', 'india-jackson-1', f'predictions_{current_datetime}.csv',
                     f'remote_vm_test/predictions/predictions_{current_datetime}.csv')

    print(f"Predictions saved to {predictions_save_path}")

    # Save inputs to CSV
    inputs_df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(data.shape[1])])
    inputs_df.to_csv(inputs_save_path, index=False)
    upload_to_gcloud('hl-geo', 'india-jackson-1', f'inputs_{current_datetime}.csv',
                     f'remote_vm_test/inputs/inputs_{current_datetime}.csv')

    print(f"Inputs saved to {inputs_save_path}")

    return predictions

if __name__ == "__main__":
    times, data = generate_sine_wave(freq=1, sample_rate=100, duration=10)
    inputs = np.array([data[i:i+10] for i in range(len(data)-10)])
    model = LSTMPredictor(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    model.load_state_dict(torch.load('lstm_model.pth'))

    current_datetime = get_current_datetime()
    predictions = predict_and_plot(model, inputs, f'inputs_{current_datetime}.csv',f'predictions_{current_datetime}.csv')
    subprocess.run(["python", "save_outputs.py"], check=True)

    plt.figure(figsize=(10, 5))
    plt.plot(times[10:], data[10:], label='Actual')
    plt.plot(times[10:], predictions, label='Predicted', linestyle='--')
    plt.title("Time Series Prediction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

