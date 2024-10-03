from io import BytesIO

import h5py
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# Add background color
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background-image: url("https://storage.googleapis.com/india-jackson-1/NEW_BANNER_2.png");
        background-size: cover;
        background-position: center;
        //background-repeat: no-repeat;
        width: 100%;
        height: 100px; /* Adjust this as needed */
        //display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    section[data-testid="stSidebar"] {
        top: 100px;
        //width: 500px;
        
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        width: 100%; //calc(100% - 350px); /* Adjust the width to use full space, considering sidebar width */
    }
    
    .box {
        background: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        width: 100%;
        height: 90vh;
        margin: auto;
        margin-top: 20px;
        padding: 20px;
        color: black;
    }
    .box:hover {
        box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
    }
    .box-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=[
                    "FLD 2024 - GeoCLoak",
                    "FDL 2023 - SHEATH",
                    "FDL 2020 - DAGGER",
                    "Data Sources",
                ],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=[
                "FLD 2024 - GeoCLoak",
                "FDL 2023 - SHEATH",
                "FDL 2020 - DAGGER",
            ],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=[
                "FLD 2024 - GeoCLoak",
                "FDL 2023 - SHEATH",
                "FDL 2020 - DAGGER",
            ],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=1)


if selected == "FLD 2024 - GeoCLoak":
    # st.title(f"{selected}")
    st.markdown(
        f"""
        <center>
            <h1>{selected}</h1>
        </center>
        """,
        unsafe_allow_html=True,
    )
    # response = requests.get("https://storage.googleapis.com/india-jackson-1/FDL_GeoCLoak_3.png")
    response = requests.get(
        "https://storage.googleapis.com/india-jackson-1/geocloak_2.png"
    )
    img = Image.open(BytesIO(response.content))
    st.image(
        img,
        # caption="FDL-Geoeffectiveness Banner",
        # width=800,
        channels="RGB",
    )

    # Add the box with depth effect
    st.markdown(
        """
        <div class="box">
            <div class="box-header">Geoeffectiveness-CL or Active Knowledge (GeoCLoak)</div>
            <!-- Add your dashboard content here -->
            <center>
            <p>This is a placeholder for the dashboard content.</p>
            </center>
        </div>
        """,
        unsafe_allow_html=True,
    )

if selected == "FDL 2023 - SHEATH":
    st.markdown(
        f"""
        <center>
            <h1>{selected}</h1>
        </center>
        """,
        unsafe_allow_html=True,
    )
    response = requests.get(
        "https://storage.googleapis.com/india-jackson-1/SHEATH_3.png"
    )
    img = Image.open(BytesIO(response.content))
    st.image(
        img,
        # caption="FDL-Geoeffectiveness Banner",
        # width=800,
        channels="RGB",
    )

    # Add the box with depth effect
    st.markdown(
        """
        <div class="box">
            <div class="box-header">Solar wind High-speed Enhancements And Transients Handler (SHEATH)</div>
            <!-- Add your dashboard content here -->
            <center>
            <p>This is a placeholder for the dashboard content.</p>
            </center>
        </div>
        """,
        unsafe_allow_html=True,
    )
if selected == "FDL 2020 - DAGGER":
    st.markdown(
        f"""
        <center>
            <h1>{selected}</h1>
        </center>
        """,
        unsafe_allow_html=True,
    )
    response = requests.get("https://storage.googleapis.com/india-jackson-1/DAGGER.png")
    img = Image.open(BytesIO(response.content))
    st.image(
        img,
        # caption="FDL-Geoeffectiveness Banner",
        # width=800,
        channels="RGB",
    )

    # Add the box with depth effect
    st.markdown(
        """
        <div class="box">
            <div class="box-header">Deep leArninG Geomagnetic pErtuRbation (DAGGER)</div>
            <!-- Add your dashboard content here -->
            <center>
            <p>This is a placeholder for the dashboard content.</p>
            </center>
        </div>
        """,
        unsafe_allow_html=True,
    )

if selected == "Data Sources":
    st.markdown(f"<center><h1>{selected}</h1></center>", unsafe_allow_html=True)

    # URL of the HDF5 file on Google Cloud Storage
    url = "https://storage.googleapis.com/india-jackson-1/formatted_data_OMNI_omniweb_formatted_2000.h5"

    def download_h5_from_gcs(url):
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    # Download the HDF5 file
    h5_data = download_h5_from_gcs(url)

    # Open the HDF5 file and extract data
    with h5py.File(h5_data, "r") as hdf:
        labels = [label.decode("utf-8") for label in hdf["data/axis0"][:]]
        timestamps = hdf["data/axis1"][:]
        data_values = hdf["data/block0_values"][:]
        timestamps = pd.to_datetime(timestamps)
        df = pd.DataFrame(data=data_values, index=timestamps, columns=labels)
        df_reset = df.reset_index().rename(columns={"index": "Timestamp"})

    # Display the DataFrame in Streamlit
    st.write("Data from HDF5 file:")
    st.dataframe(df_reset)
    # Make sure to give credit for the data sources
