import os

import h5py
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

# from data_module import fetch_data, prepare_dataframe, save_csv, plot_time_series
from data_module import fetch_data, prepare_dataframe, save_csv

token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

# Define constants for bucket and measurement names
buckets = {"ACE": "ace_bucket", "DSCOVR": "dscovr_bucket"}
measurements = {"ACE": "ace_data", "DSCOVR": "dscovr_data"}

"""''
def plot_time_series(df):
    # Create a plotly figure
    fig = go.Figure()

    # Add traces for each field in the DataFrame
    for column in df.columns:
        if column != 'time':  # Assuming 'time' is the time index
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df[column],
                mode='lines',
                name=column
            ))

    # Update layout for a dark theme
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode='x unified',  # Unified hover to show all field values together
        title='Time Series Data',
        xaxis_title='Time',
        yaxis_title='Value'
    )

    # Update axes to match the dark theme
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='white')
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='white')

    return fig
"""

import plotly.graph_objs as go


def plot_time_series(df):
    # Ensure that the 'time' column is present
    if "time" not in df.columns:
        # Check if the time data is in the index and reset it
        df = df.reset_index()

    # Create a plotly figure
    fig = go.Figure()

    # Add traces for each field in the DataFrame, except the 'time' column
    for column in df.columns:
        if column != "Time":  # Update 'time' to the actual name of your time column
            fig.add_trace(
                go.Scatter(
                    x=df[
                        "Time"
                    ],  # Replace 'time' with the correct column name if necessary
                    y=df[column],
                    mode="lines",
                    name=column,
                )
            )

    # Update layout for a dark theme
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hovermode="x unified",  # Unified hover to show all field values together
        title="Time Series Data",
        xaxis_title="Time",
        yaxis_title="Value",
    )

    # Update axes to match the dark theme
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="white")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="white")

    return fig


def display_data(bucket, measurement, csv_file, title):
    # Fetch and display data
    with st.spinner(f"Fetching {measurement} data..."):
        result = fetch_data(token, org, url, bucket, measurement)
        df = prepare_dataframe(result)

    # Display DataFrame
    st.write(f"### {title}")
    st.write(df)

    # Generate and display plots
    fig = plot_time_series(df)
    st.plotly_chart(fig)
    # for fig in figures:
    # st.pyplot(fig)

    # Save CSV and provide download button
    save_csv(df, csv_file)
    st.download_button(
        label=f"Download {title} Data as CSV",
        data=open(csv_file, "rb"),
        file_name=csv_file,
        mime="text/csv",
    )


def get_data():

    # Split the page into two columns for ACE and DSCOVR
    col1, col2 = st.columns([1, 1])

    with col1:
        display_data(buckets["ACE"], measurements["ACE"], "ace_data.csv", "ACE Data")

    with col2:
        display_data(
            buckets["DSCOVR"], measurements["DSCOVR"], "dscovr_data.csv", "DSCOVR Data"
        )
