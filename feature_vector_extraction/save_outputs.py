import torch
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
from train_model import LSTMPredictor, generate_sine_wave
import datetime


def get_current_datetime():
    # Returns a formatted datetime string, e.g., '20230901_150505'
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

'''''
def load_model_and_save_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1).to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    # Generate some test input data (e.g., a sine wave segment)
    _, data = generate_sine_wave(freq=1, sample_rate=100, duration=2)  # Generate 2 seconds of sine wave
    input_tensor = torch.tensor(data).float().unsqueeze(0).unsqueeze(-1).to(
        device)  # Reshape for LSTM [Batch, Sequence, Features]

    with torch.no_grad():
        output = model(input_tensor)

    # Plotting the actual input and the model's prediction
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Actual Sine Wave')
    plt.plot(output.cpu().numpy().flatten(), label='Predicted', linestyle='--')
    plt.legend()
    plt.title("LSTM Time Series Prediction")
    plt.xlabel("Time steps")
    plt.ylabel("Amplitude")
    plt.savefig('output_time_series.png')  # Save the figure as a PNG file

    # Initialize Cloud Storage and upload the file
    storage_client = storage.Client(project='hl-geo')
    bucket = storage_client.bucket('india-jackson-1')
    blob = bucket.blob('remote_vm_test/output_time_series.png')
    blob.upload_from_filename('output_time_series.png')
    print("Output saved to Google Cloud Storage.")


if __name__ == "__main__":
    load_model_and_save_output()
'''


def upload_to_gcloud(project_name, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def load_model_and_save_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1).to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    # Generate some test input data (e.g., a sine wave segment)
    _, data = generate_sine_wave(freq=1, sample_rate=100, duration=2)
    input_tensor = torch.tensor(data).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    current_datetime = get_current_datetime()

    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Actual Sine Wave')
    plt.plot(output.cpu().numpy().flatten(), label='Predicted', linestyle='--')
    plt.legend()
    plt.title("LSTM Time Series Prediction")
    plt.xlabel("Time steps")
    plt.ylabel("Amplitude")
    plt.savefig(f'time_series_{current_datetime}.png')


    # Upload files to Google Cloud Storage
    upload_to_gcloud('hl-geo', 'india-jackson-1', f'time_series_{current_datetime}.png', f'remote_vm_test/graphs/time_series_{current_datetime}.png')


if __name__ == "__main__":
    load_model_and_save_output()