"""
The configuration data for NRT data.
"""

# Faradey Cup data
dscovr_f1m_cols = ["proton_speed", "proton_density", "proton_temperature"]


# Mag data
dscovr_m1m_cols = ["bt", "bx_gsm", "by_gsm", "bz_gsm"]

column_names = {
    "proton_speed": "Speed",
    "proton_density": "Density",
    "proton_temperature": "Temperature",
    "bt": "Bt",
    "bx_gsm": "Bx",
    "by_gsm": "By",
    "bz_gsm": "Bz",
}
