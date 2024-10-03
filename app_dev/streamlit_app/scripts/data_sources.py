import os

# from data_module import fetch_data, prepare_dataframe, save_csv, plot_time_series
from datetime import datetime

import h5py
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from data_module import fetch_data, prepare_dataframe, save_csv

token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

# Define constants for bucket and measurement names
buckets = {"ACE": "ace_bucket", "DSCOVR": "dscovr_bucket"}
measurements = {"ACE": "ace_data", "DSCOVR": "dscovr_data"}


def plot_time_series(df, selected_fields):
    # Ensure that the 'time' column is present
    if "Time" not in df.columns:
        df = df.reset_index()

    # Create a plotly figure with an initially blank graph
    fig = go.Figure()

    # Add traces for each selected field
    for column in selected_fields:
        fig.add_trace(go.Scatter(x=df["Time"], y=df[column], mode="lines", name=column))

    # Update layout for a dark theme
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hovermode="x unified",
        title="Time Series Data",
        xaxis_title="Time",
        yaxis_title="Value",
    )

    # Update axes to match the dark theme
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="white")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="white")

    return fig


def display_data(bucket, measurement, csv_file, title):
    st.write(f"### {title}")
    # Fetch and display data
    with st.spinner(f"Fetching {measurement} data..."):
        result = fetch_data(token, org, url, bucket, measurement)
        df = prepare_dataframe(result)

    selected_fields = []
    for field in df.columns:
        if field != "time":
            if st.checkbox(f"{field}", key=f"{title}_{field}"):
                selected_fields.append(field)

    # Display the plot dynamically as fields are selected
    fig = plot_time_series(df, selected_fields)
    st.plotly_chart(fig)

    st.write(df)
    # Always display download button
    save_csv(df, csv_file)
    st.download_button(
        label=f"Download {title} Data as CSV",
        data=open(csv_file, "rb"),
        file_name=csv_file,
        mime="text/csv",
    )


def get_data():

    col1, col2 = st.columns([1, 1])

    with col1:
        # Display ACE data for the selected date
        display_data(buckets["ACE"], measurements["ACE"], "ace_data.csv", f"ACE Data")

    with col2:
        # Display DSCOVR data for the selected date
        display_data(
            buckets["DSCOVR"], measurements["DSCOVR"], "dscovr_data.csv", f"DSCOVR Data"
        )
