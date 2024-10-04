import logging

import functions_framework
from indicesdata import collect_fadj_data, collect_hp_ap_data, collect_kp_data

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@functions_framework.http
def main(request):
    # Check for 'index' in query parameters
    index = request.args.get("index")

    # If 'index' is not in query parameters, check the JSON body
    if not index:
        request_json = request.get_json(silent=True)
        if request_json and "index" in request_json:
            index = request_json["index"]

    if not index:
        log.error("Index parameter is missing")
        return "Index parameter is missing", 400

    print(index)

    if index == "Kp":
        log.info(f"Collecting data for index {index}")
        collect_kp_data()
    elif index in ["Hp30", "ap30"]:
        log.info("Collecting data for indices Hp30 and ap30")
        collect_hp_ap_data()
    elif index == "Fadj":
        log.info(f"Collecting data for index {index}")
        collect_fadj_data()
    else:
        log.error(f"Invalid index parameter: {index}")
        return f"Invalid index parameter: {index}", 400

    log.info("Data collection complete.")
    return "Data collection complete.", 200
