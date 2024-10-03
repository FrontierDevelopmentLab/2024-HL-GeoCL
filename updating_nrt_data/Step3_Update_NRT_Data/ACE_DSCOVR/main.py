import logging

import functions_framework
from flask import jsonify
from nrtdata import NRTData


@functions_framework.http
def main(request):
    nrt_data = NRTData()

    # Initialize a dictionary to gather responses
    response_data = {}

    ace_data = nrt_data.nrtACE
    if ace_data is not None:
        print("ACE Data:")
        print(ace_data.head())
        nrt_data.save_to_influxdb(ace_data, "ace_data", "ace_bucket")
        response_data["ace_data"] = "Data processed and saved to ACE bucket."
    else:
        response_data["ace_data"] = "No ACE data available."
    dscovr_data = nrt_data.nrtDSCOVER
    if dscovr_data is not None:
        print("DSCOVR Data:")
        print(dscovr_data.head())
        nrt_data.save_to_influxdb(dscovr_data, "dscovr_data", "dscovr_bucket")
        response_data["dscovr_data"] = "Data processed and saved to DSCOVR bucket."
    else:
        response_data["dscovr_data"] = "No DSCOVR data available."

    # Return a JSON response with the status of data processing
    return jsonify(response_data), 200
