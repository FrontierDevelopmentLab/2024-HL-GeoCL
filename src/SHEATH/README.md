# SHEATH: Solar wind and High speed Ehancements And Transients Handler

This module takes in the SDOML dataset and performs solar wind and IMF forecasting at L1. 

The module has a training setup and an inference setup, which are handled separately, since we have the full dataset available for training in a disk.

## Training setup
We first download hourly data from OMNI: [OMNI hourly data](https://omniweb.gsfc.nasa.gov/form/dx1.html), and preprocess the dataset. The OMNI data is backtraced, and the AIA data closest to the backtraced times are taken up. We then generate the CH-AR mask from AIA 193, and then generate the feature set for the full dataset. 
We consider a timeframe of AIA data, query and preprocess them, and then perform the inference. The forecasts are saved in the format expected by DAGGER.
1. Clean OMNI data by removing Nans, and converting the text files into hdf5 files: ```bash python preprocess_omni/preprocess_omni_hourly.py```. This generates the file `omni_preprocess_1hour_full.h5`, which is saved at `/home/jupyter/Vishal/omni/` in our local instance. . This forms our **target data**.
2. Backtrace OMNI data: We need to backtrace the OMNI data in ```bash python preprocess_omni/backtrace_omni_aia.py```. This will generate dates which will be used to select the AIA data. We can use either radial backtracing (using `get_backtrace_date`) or spiral backtracing (using `backtrace_spiral_date`). This generates `sun_backtraced_dates.npy` in `logs/`, which contains the AIA dates nearest to the source regions of the solar wind measurements in the OMNI dataset.
3. Train-test-val split: Since we are basing our data time series on the OMNI data, we perform the train-test-val split in ```bash python preprocess_omni/select_dates.py```. These dates are saved in `sheath_train_test_val_split.npz` in `logs/`. 
4. We now select the AIA 193 A dates corresponding OMNI backtraced dates. This is done in ```bash python preprocess_aia/aia_193_inds_for_omni.py```. We generate these dates in `closest_omni_aia_dates_{year}.npy` for different dates, and the indices in `closest_omni_aia_indices_{year}.npy`. These are saved in `logs/`.
5. Generate the CH and AR mask using the AIA 193 A dataset: ```bash python preprocess_aia/Generate_central_mask_CH.py```. This generates the masks at `sheath_aia_data/ch_mask_{year}.npy` and the corresponding AIA 193 A timestamps at `sheath_aia_data/AIA193_times_{year}.npy`. 
6. The masks are now folded multiplied with the AIA and HMI data next in ```bash python preprocess_aia/aia_combine_all_data.py```. This generates a set of files at `sheath_aia_data/aia_subsamp_masked_{year}.npy`, which contain the intensity mask in 193, and the intensities in CH and AR in different AIA and HMI passbands. It also takes their average spatially to generate the fractional intensities and masks as `sheath_aia_data/aia_subsamp_masked_summed_{year}.npy`. This forms our **input data**.
7. Training script: The model is trained in ```bash python XGBoost_model.py```. It generates the model checkpoints and results in `logs/` as `SHEATH_xgb.model`.

## SHEATH Inference setup

### Online inference
The main inference setup is an online query of SDOML dataset, and further preprocessing, etc. 


### Offline inference


### DAGGER inference
With the SHEATH forecasts, we then just feed in the data into DAGGER model for performing geomag forecasting!