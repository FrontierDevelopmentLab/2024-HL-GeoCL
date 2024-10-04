from io import BytesIO

import requests
import streamlit as st
from PIL import Image


def get_geocloak():
    # st.title(f"{selected}")
    st.html(
        """
        <center>
            <h1>GeoCloak</h1>
        </center>
        """,
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
    st.html(
        """
        <div class="box">
            <div class="box-header">Geoeffectiveness-CL or Active Knowledge (GeoCLoak)</div>
            <!-- Add your dashboard content here -->
            <center>
            <p>This is a placeholder for the dashboard content.</p>
            </center>
        </div>
        """,
    )
