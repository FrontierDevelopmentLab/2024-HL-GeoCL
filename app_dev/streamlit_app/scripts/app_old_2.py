from io import BytesIO

import requests
import streamlit as st
from data_module import fetch_data, plot_time_series, prepare_dataframe, save_csv
from PIL import Image
from streamlit_option_menu import option_menu

token = "8xj_AxxKhgKYz9oMfrsgyacXB-b8BgvG-KpBtKLZIK3yvQr7v7MCZmNQFOmyJ0N56m_OEnY1Snwn4O86foraEQ=="
org = "Google"
url = "https://34.48.13.92:8086"

st.html(
    """
    <style>
    header[data-testid="stHeader"] {
        background-image: url("https://storage.googleapis.com/india-jackson-1/NEW_BANNER_2.png");
        background-size: cover;
        background-position: center;
        width: 100%;
        height: 100px; /* Adjust this as needed */
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    section[data-testid="stSidebar"] {
        top: 100px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        width: 100%;
    }

    section.main.st-emotion-cache-bm2z3a.ea3mdgi8 {
        padding-right: 700px;  Adjust the padding as needed */
    }

    .box {
        background: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        width: 90vh;
        height: 100%;
        margin: 20px auto; /* Center the box horizontally and add top margin */
        align-items: center;
        color: black;
        padding: 20px;  /*Add padding inside the box */
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
    # unsafe_allow_html=True,
)


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=[
                    "About",
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

if selected == "About":
    st.html(
        f"""
        <center>
            <h1>{selected}</h1>
        </center>
        """,
        # unsafe_allow_html=True,
    )

    # Geoeffectiveness Section
    st.html(
        """
        <div class = "box">
            <div style="border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: black;">
                <div class="box-header">Geoeffectiveness</div>
                <p>Geoeffectiveness refers to the impact of solar and geomagnetic activity on Earth's environment and technological systems. Understanding and predicting geoeffective events, such as geomagnetic storms, is crucial for mitigating their effects on satellites, power grids, and communication systems.</p>
                <p>Frontier Development Lab (FDL) contributes to this field using advanced Machine Learning (ML) and Artificial Intelligence (AI) techniques to enhance our understanding and prediction capabilities, ensuring better preparedness for geoeffective events. Frontier Development Lab (FDL) contributes to this field using advanced Machine Learning (ML) and Artificial Intelligence (AI) techniques to enhance our understanding and prediction capabilities, ensuring better preparedness for geoeffective events. Through ML models like the Solar wind High-speed Enhancements And Transients Handler (SHEATH) and the Deep leArninG Geomagnetic pErtuRbation (DAGGER), FDL is able to identify, analyze, and forecast solar wind transients and geomagnetic perturbations with unprecedented accuracy, providing critical insights that help mitigate the risks associated with space weather events.</p>

                <img src="https://storage.googleapis.com/india-jackson-1/sheath_dagger.png" style="width:100%; height:auto; margin-bottom: 20px;">

                <p>The Solar wind High-speed Enhancements And Transients Handler (SHEATH) project, initiated during FDL 2020, focuses on identifying and analyzing high-speed solar wind streams and their interactions with Earth's magnetosphere. This project aims to enhance our understanding of solar wind dynamics and their impact on geomagnetic activity. Following SHEATH, the Deep leArninG Geomagnetic pErtuRbation (DAGGER) project was developed during FDL 2023. It utilizes deep learning techniques to predict geomagnetic disturbances caused by solar wind and other space weather phenomena. The objective of DAGGER is to improve space weather forecasting capabilities and provide timely alerts for potential geoeffective events. Building on the progress of SHEATH and DAGGER, the Continuous Learning (CL) initiative integrates real-time data acquisition with machine learning to continually refine model accuracy and predictive power. Effective implementation of CL depends on the availability of Near Real-Time (NRT) data, which is crucial for updating models with the latest observations and enhancing forecast reliability..</p>

            </div>

            <div style="border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: black;">

            <div class="box-header">Near Real Time (NRT) Data</div>
                <p>Near Real-Time (NRT) data refers to data that is collected and processed with minimal delay, allowing for real-time analysis and decision-making. NRT data is crucial for the success of Continuous Learning, as it provides the most up-to-date information available. Near Real-Time Solar Dynamic Observatory Machine Learning (NRT SDOML) involves using real-time data from the Solar Dynamic Observatory (SDO) to train and update machine learning models. This approach helps in monitoring and predicting solar activities that can impact space weather. L1 data refers to data collected from the first Lagrangian point (L1), where the gravitational forces of the Earth and the Sun balance the orbital motion of a satellite. This makes L1 an ideal location for monitoring solar wind and other space weather phenomena.</p>

                <ul>
                    <li><strong>SDO:</strong> The Solar Dynamics Observatory (SDO) spacecraft, operated by NASA, continuously observes the Sun and sends its Level 0 (L0) data to ground stations like the White Sands Complex in New Mexico. From there, the data is transmitted to NASA’s Goddard Space Flight Center (GSFC), where it is ingested into the SDO Data Processing Center. Once the L0 data is received at GSFC, it undergoes processing to generate higher-level data products, such as Level 1 and Level 2 data. These products include detailed measurements of the Sun’s magnetic field, solar irradiance, and images of the Sun in various ultraviolet wavelengths. The processed data is then made available to the scientific community and the public through the Joint Science Operations Center (JSOC) and other NASA data repositories, contributing to a deeper understanding of solar dynamics and space weather forecasting.</li>

                    <li><strong>ACE:</strong> The Advanced Composition Explorer (ACE) is located at L1, collecting solar wind data, including speed, density, and magnetic field strength. ACE spacecraft sends its Level 0 (L0) data to the Deep Space Network (DSN), which is managed by NASA's Jet Propulsion Laboratory (JPL). The DSN is a network of large antennas and communication facilities located around the world, which are used to communicate with spacecraft that are beyond low Earth orbit. Once the L0 data is received by the DSN, it is sent to the Space Physics Data Facility (SPDF) at NASA's Goddard Space Flight Center. The data is then processed into higher-level data products (e.g., Level 1 and Level 2 data) and made available to the scientific community through the SPDF and other NASA data centers. The data is downlinked in near real-time to Earth, where it is processed and ingested into various space weather models.</li>

                    <li><strong>DSCOVR:</strong> The Deep Space Climate Observatory (DSCOVR) is also located at L1, providing critical data on solar wind and magnetic fields. Like ACE, DSCOVR’s data is downlinked in near real-time and used to monitor and predict space weather conditions. DSCOVR (Deep Space Climate Observatory) spacecraft sends its Level 0 (L0) data to NOAA’s Satellite Operations Facility (NSOF) after it is received by ground stations, such as those at the NOAA Wallops Command and Data Acquisition Station (WCDAS) in Virginia. The data is then transmitted to NOAA's Space Weather Prediction Center (SWPC) for processing. Once the L0 data is received at NSOF, it is processed into higher-level data products, including Level 1 and Level 2 data, which provide detailed information on solar wind parameters and magnetic field measurements. These processed data products are made available to the scientific community and the public through NOAA's data repositories and other relevant platforms.</li>

                    <li><strong>Geomagnetic Indices (Hp30, ap30, Kp):</strong> These indices are collected by various instruments located on Earth. They measure geomagnetic activity, providing essential information on how solar wind interacts with Earth's magnetic field. This data is crucial for understanding the impact of solar events on Earth's environment. Geomagnetic indices like Kp, ap, and Hp30 are vital tools for monitoring and predicting space weather events that can impact Earth's technological systems. These indices are derived from data collected by a global network of ground-based magnetometers strategically located to provide comprehensive coverage of Earth's magnetic field variations. The Kp index, for instance, is calculated using measurements from 13 key geomagnetic observatories around the world. These observatories, such as Niemegk in Germany, Førde in Norway, Hartland in the UK, Wingst in Germany, and Eskdalemuir in the UK, among others, provide a consistent and global measure of geomagnetic activity. The ap index, which is a linear equivalent of the Kp index, also relies on data from these observatories to offer a more detailed view of geomagnetic disturbances in nanoteslas (nT). High-latitude geomagnetic activity, which is crucial for understanding auroral activity and polar disturbances, is captured by the Hp30 index. This index is derived from magnetometer data collected at observatories in high-latitude regions such as Qaanaaq in Greenland, Tromsø in Norway, Barrow in the USA, Vostok in Antarctica, and McMurdo in Antarctica. These stations are ideally situated to monitor the intense geomagnetic activity often seen in these regions. In addition to these indices, other specialized indices like Dst (Disturbance Storm Time) and AE (Auroral Electrojet) are generated from different sets of ground-based magnetometers, focusing on specific latitudes or regions to provide insights into storm-time disturbances and auroral zone activities. The data from these observatories are collected, processed, and distributed by national and international data centers, such as NOAA’s Space Weather Prediction Center (SWPC) and the World Data Centers for Geomagnetism, playing a critical role in space weather forecasting and research.</li>
                </ul>
            </div>

            <div style="border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: black;">
                <!--<div class="box-header">FDL-Geo Teams</div>-->
                    <div style="display: grid; grid-template-columns: 1fr; gap: 20px;">

                        <div class="box-header">FDL-2024 GeoCloak</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/geocloak_2.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <div class="box-header">FDL-2024 SHEATH</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/SHEATH_3.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <div class="box-header">FDL-2024 DAGGER</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/DAGGER.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <!--
                        <div class="box-header">FDL-2025 <new_team_name></div>
                        <div><img src="path/to/team_image_4.png" style="width:100%; height:auto;"></div>-->
                        <p></p>
                        -->

                    </div>
                </div>

        </div>

        """,
        # unsafe_allow_html=True,
    )


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

# Define constants for bucket and measurement names
buckets = {"ACE": "ace_bucket", "DSCOVR": "dscovr_bucket"}
measurements = {"ACE": "ace_data", "DSCOVR": "dscovr_data"}

# Display section title
if selected == "Data Sources":
    st.markdown(f"<center><h1>{selected}</h1></center>", unsafe_allow_html=True)

    # Create buttons for selecting data source
    st.markdown(
        """
                <style>
                    div[data-testid="column"] {
                        width: fit-content !important;
                        flex: unset;
                    }
                    div[data-testid="column"] * {
                        width: fit-content !important;
                    }
                </style>
                """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ace_button = st.button("ACE")
    with col2:
        dscovr_button = st.button("DSCOVR")

    # Determine which button is clicked and set parameters
    if ace_button:
        bucket = buckets["ACE"]
        measurement = measurements["ACE"]
        csv_file = "ace_data.csv"
    elif dscovr_button:
        bucket = buckets["DSCOVR"]
        measurement = measurements["DSCOVR"]
        csv_file = "dscovr_data.csv"
    else:
        # Default or fallback scenario if no button is clicked yet
        st.stop()

    # Fetch and display data for the selected source
    with st.spinner(f"Fetching {measurement} data..."):
        result = fetch_data(token, org, url, bucket, measurement)
        df = prepare_dataframe(result)

    # Display DataFrame
    st.write(df)

    # Generate and display plots
    figures = plot_time_series(df)
    for fig in figures:
        st.pyplot(fig)

    # Download CSV functionality
    save_csv(df, csv_file)
    st.download_button(
        label="Download Data as CSV",
        data=open(csv_file, "rb"),
        file_name=csv_file,
        mime="text/csv",
    )
