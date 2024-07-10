#!/bin/bash
python Generate_central_mask_CH.py
cd ../
gsutil -m cp -r sheath_aia_data gs://us-fdlx-geo-cme/