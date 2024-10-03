import pandas as pd
import streamlit as st
from data_module import fetch_data, plot_time_series, prepare_dataframe, save_csv

# Constants for InfluxDB connection
token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

# Define constants for bucket and measurement names
buckets = {"ACE": "ace_bucket", "DSCOVR": "dscovr_bucket"}
measurements = {"ACE": "ace_data", "DSCOVR": "dscovr_data"}


def display_data_html(bucket, measurement, csv_file, title):
    # Fetch and display data
    with st.spinner(f"Fetching {measurement} data..."):
        result = fetch_data(token, org, url, bucket, measurement)
        df = prepare_dataframe(result)

    # Convert DataFrame to HTML table
    df_html = df.to_html(classes="data", index=False)

    # Generate plots
    figures = plot_time_series(df)
    plots_html = ""
    for fig in figures:
        plots_html += st.pyplot(fig, use_container_width=True)

    # Save CSV for download
    save_csv(df, csv_file)
    download_link = f'<a href="data:file/csv;base64,{st.download_button(csv_file)}" download="{csv_file}">Download {title} Data as CSV</a>'

    return f"""
    <div style="text-align: justify; border: 2px solid #ffffff; border-radius: 10px; 
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; 
                width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: black;">
        <h2>{title}</h2>
        {df_html}
        {plots_html}
        {download_link}
    </div>
    """


def get_data():
    st.html(
        f"<center><h1>DATA SOURCES</h1></center>",
    )

    # Start the main box
    st.html(
        """
        <div class="box" style="display: flex; justify-content: space-between;">
    """,
    )

    # ACE Data
    ace_html = display_data_html(
        buckets["ACE"], measurements["ACE"], "ace_data.csv", "ACE Data"
    )
    st.html(
        f"<div style='flex: 1; margin-right: 10px;'>{ace_html}</div>",
    )

    # DSCOVR Data
    dscovr_html = display_data_html(
        buckets["DSCOVR"], measurements["DSCOVR"], "dscovr_data.csv", "DSCOVR Data"
    )
    st.html(
        f"<div style='flex: 1; margin-left: 10px;'>{dscovr_html}</div>",
    )

    # Close the main box
    st.html(
        "</div>",
    )


if __name__ == "__main__":
    get_data()
