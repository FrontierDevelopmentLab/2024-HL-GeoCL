import streamlit as st
from style import render_custom_style
from menu_options import streamlit_menu
from about_content import render_about_content
from sheath import get_sheath
from dagger import get_dagger
from geocloak import get_geocloak
from data_sources import get_data

# Set page configuration
st.set_page_config(page_title="FDL-Geo", page_icon=None, layout='wide', initial_sidebar_state='auto')

render_custom_style()

selected = streamlit_menu(example=1)

if selected == "About":
    render_about_content()


if selected == "GeoCLoak":
    get_geocloak()

if selected == "SHEATH":
    get_sheath()

if selected == "DAGGER-CL":
    get_dagger()

# Display section title
if selected == "Data Sources":
    get_data()


