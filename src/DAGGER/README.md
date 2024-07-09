# Running the codes.

1. Install all dependencies as `pip install -r requirements.txt`.
2. Ensure the data is downloaded by running the bash script `get_data.sh` in `scripts/`. 
3. Move the data to the main repo as `mv data_local ../`
4. Run `preprocess.py`, with appropriate path for including regularization or not (and update `train_script.py` with appropriate preprocessed data output folder).
5. If additional station regularization files are needed, run `scripts/station_regularization.py`.
6. Then, for running the parameter sweep, do: `wandb sweep sweep.yaml`, which will give you a sweep_id (actually a command as wandb agent ...)
7. Run the command `wandb agent <sweep_id>`

Use `train_script.py`, not `train.py`. The latter is a prototyping version.

Make sure you `unset LD_LIBRARY_PATH` as needed.
Also, please remember to put this tip into a final README. :)

Reference for the sweeps: https://github.com/borisdayma/lightning-kitti. 

The data and model checkpoints are in "geo2020_supermag/Uniformdata_multiyear/" in the bucket.