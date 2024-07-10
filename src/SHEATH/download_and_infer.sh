gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_small.zarr ../sheath_data/sdoml_data/
gsutil -m cp -r gs://us-fdlx-landing/fdl-sdoml-v2/sdomlv2_hmi_small.zarr ../sheath_data/sdoml_data/

python3 preprocessing/Generate_central_mask_CH.py
python3 preprocessing/Generate_ch_summaries.py

python3 SHEATH_predict.py
