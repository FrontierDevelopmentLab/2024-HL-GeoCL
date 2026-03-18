import datetime

import requests
import streamlit as st
from google.cloud import storage


def get_last_uploaded_object(project_name, bucket_name, prefix):
    """Retrieve the most recently uploaded object in a specified 'directory' within a bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # List all objects that have the prefix

    # Initialize a variable to track the most recent object
    last_uploaded = None
    last_uploaded_time = None

    for blob in blobs:
        if last_uploaded_time is None or blob.time_created > last_uploaded_time:
            last_uploaded = blob.name
            last_uploaded_time = blob.time_created

    return last_uploaded, last_uploaded_time


def extract_timestamp_from_filename(filename):
    """Extract and format the timestamp from the filename."""
    # Assuming the timestamp is the last part of the filename before the extension
    timestamp_str = filename.split("/")[-1].split(".")[0][
        -12:
    ]  # Extract the timestamp part
    timestamp = datetime.datetime.strptime(
        timestamp_str, "%Y%m%d%H%M"
    )  # Parse the timestamp
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M")  # Format the timestamp
    return formatted_timestamp


def get_dagger():

    last_uploaded_prediction, last_uploaded_prediction_time = get_last_uploaded_object(
        "hl-geo", "india-jackson-1", "uncertainty_vm_test/predictions"
    )
    last_uploaded_input, last_uploaded_input_time = get_last_uploaded_object(
        "hl-geo", "india-jackson-1", "uncertainty_vm_test/inputs"
    )

    last_uploaded_dbe_mean_graph, last_uploaded_dbe_mean_graph_time = (
        get_last_uploaded_object(
            "hl-geo", "india-jackson-1", "uncertainty_vm_test/graphs/dbe_geo_mean"
        )
    )
    last_uploaded_dbn_mean_graph, last_uploaded_dbn_mean_graph_time = (
        get_last_uploaded_object(
            "hl-geo", "india-jackson-1", "uncertainty_vm_test/graphs/dbn_geo_mean"
        )
    )
    last_uploaded_dbe_sd_graph, last_uploaded_dbe_sd_graph_time = (
        get_last_uploaded_object(
            "hl-geo", "india-jackson-1", "uncertainty_vm_test/graphs/dbe_geo_sd"
        )
    )
    last_uploaded_dbn_sd_graph, last_uploaded_dbn_sd_graph_time = (
        get_last_uploaded_object(
            "hl-geo", "india-jackson-1", "uncertainty_vm_test/graphs/dbn_geo_sd"
        )
    )

    inputs_data = requests.get(
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_input}",
        allow_redirects=True,
    )
    predictions_data = requests.get(
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_prediction}",
        allow_redirects=True,
    )

    # Define the URL for each image
    image_urls = [
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_dbe_mean_graph}",
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_dbn_mean_graph}",
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_dbe_sd_graph}",
        f"https://storage.googleapis.com/india-jackson-1/{last_uploaded_dbn_sd_graph}",
    ]

    """
    image_urls = [
        "https://storage.googleapis.com/india-jackson-1/remote_vm_test/graphs/secgp_mean_Be_1.png",
        "https://storage.googleapis.com/india-jackson-1/remote_vm_test/graphs/secgp_mean_Bn.png"
        "https://storage.googleapis.com/india-jackson-1/remote_vm_test/graphs/secgp_sd_Be.png"
        "https://storage.googleapis.com/india-jackson-1/remote_vm_test/graphs/secgp_sd_Bn.png"
    ]
    """
    formatted_timestamp = extract_timestamp_from_filename(last_uploaded_dbe_mean_graph)

    st.html("""
        <div class = "box">
            <div style="background-color: black; text-align: justify; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: white;">
                <div class="box-header">Geomagnetic Pertubations</div>
                 <p style="color:white; font-size:16px;">The plots provided show two graphs that refer to the magnetic field components in a geographic coordinate system. The ground stations are represented by the yellow and red dots on the maps.</p>

                <p style="color:white; font-size:16px;"><strong>Be (East Component of Magnetic Field):</strong> This component refers to the magnetic field strength in the eastward direction. The left-hand graphs illustrate the mean and standard deviation of the eastward magnetic field component across different geographic locations.</p>

                <p style="color:white; font-size:16px;"><strong>Bn (North Component of Magnetic Field):</strong> This component refers to the magnetic field strength in the northward direction. The right-hand graphs illustrate the mean and standard deviation of the northward magnetic field component across different geographic locations.</p>

                <p style="color:white; font-size:16px;">The yellow and red dots on the maps represent the ground stations or observational points where these measurements were taken. The color gradient on the map provides a visual representation of the magnetic field strength in nanoteslas (nT) for the respective components.</p>

            </div>
        </div>""")

    # Create a 2x2 grid for images
    col1, col2 = st.columns(2)
    with col1:
        st.write("East Component of Magnetic Field")  # Add your text here
        st.image(image_urls[0], caption=None)
        st.image(image_urls[2], caption=f"{formatted_timestamp}")

    with col2:
        st.write("North Component of Magnetic Field")
        st.image(image_urls[1], caption=None)
        st.image(image_urls[3], caption=f"{formatted_timestamp}")

    # Usage example:

    print(
        f"The most recently uploaded object is: {last_uploaded_prediction} at {last_uploaded_prediction_time}"
    )
    print(
        f"The most recently uploaded object is: {last_uploaded_input} at {last_uploaded_input_time}"
    )

    # Add download buttons for the CSV files
    st.download_button(
        label="Download Inputs",
        data=inputs_data.content,
        file_name="inputs.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Predictions",
        data=predictions_data.content,
        file_name="predictions.csv",
        mime="text/csv",
    )
