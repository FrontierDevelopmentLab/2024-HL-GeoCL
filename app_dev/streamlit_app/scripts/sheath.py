import streamlit as st
from PIL import Image
import requests
from io import BytesIO


def get_sheath():
    st.html(
        f"""
        <center>
            <h1>SHEATH</h1>
        </center>
        """,
    )
    response = requests.get("https://storage.googleapis.com/india-jackson-1/SHEATH_3.png")
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
            <div class="box-header">Solar wind High-speed Enhancements And Transients Handler (SHEATH)</div>
            <!-- Add your dashboard content here -->
            <center>
            <p>This is a placeholder for the dashboard content.</p>
            </center>
        </div>
        """,
    )