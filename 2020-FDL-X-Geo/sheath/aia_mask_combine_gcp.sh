#!/bin/bash

python aia_combine_all_data.py
gsutil -m cp -r logs gs://us-fdlx-geo-cme/
gsutil -m cp -r sheath_aia_data gs://us-fdlx-geo-cme/