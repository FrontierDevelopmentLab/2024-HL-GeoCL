# 2024-HL-GeoCL
## FDL-X Heliolab 2024: Geoeffectiveness Continuous Learning

GEO-CLoak (GEOeffectiveness Continual Learning Or Active Knowledge) is an end-to-end ML pipeline for forecasting geomagnetic perturbations. It combines two models to produce predictions at different lead times:

- **SHEATH**: Predicts solar wind parameters (speed, density, temperature, magnetic field) from SDO solar imagery features. Uses a 3-layer MLP. Provides multi-day lead time, lower-fidelity forecasts.
- **DAGGER-CL**: Predicts ground-level geomagnetic field perturbations (dBe, dBn, dBz) at ~175 SuperMAG magnetometer stations. Uses a GRU encoder with continual learning (Elastic Weight Consolidation) to adapt to new data without forgetting. Provides ~30-minute lead time, higher-fidelity forecasts.

The pipeline flows: **Sun (SDO imagery) → SHEATH → solar wind forecast → DAGGER-CL (with real-time L1 data) → station predictions → global maps → web app**

## Project Structure

```
geocloak/              Main Python package
  sheath2024/          SHEATH model, dataloader, training
  dagger-cl/           DAGGER-CL model, dataloaders, inference, continual learning
  gp/                  Gaussian process & spherical harmonic interpolation
  preprocess/          SDO image preprocessing (coronal hole/active region segmentation)
  datautilus/           Real-time data download from NOAA SWPC

public/                Standalone scripts for public use
  inference_sheath.py  Run SHEATH inference from the command line
  example_data/        Sample input files

feature_vector_extraction/   Feature engineering pipeline (InfluxDB → 26 SHEATH features)
updating_nrt_data/           Near-real-time data collection (ACE, DSCOVR, geomagnetic indices)
app_dev/                     Streamlit web application
scripts/                     Training scripts, utilities, and Jupyter notebooks
models/                      Trained model checkpoints and scaler files
```

## Setup

This project requires Python 3.10+ (< 3.13) and uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

If you don't have uv installed:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

### Create environment and install dependencies

```bash
uv venv --python 3.11
uv sync
```

This creates a `.venv/` virtual environment and installs all dependencies from `pyproject.toml`.

### Activate the environment

```bash
source .venv/bin/activate
```

Or run commands directly without activating:
```bash
uv run python your_script.py
```

### Install dev tools (optional)

To also install code formatting and linting tools (black, isort, flake8):
```bash
uv sync --extra dev
```

## Contributing

The project uses black for code formatting, isort for import sorting, and flake8 for linting. Run these before submitting a pull request:
```bash
black .
isort --profile black .
flake8 --extend-ignore E203,E501
```

If you deem any lines of code correct and want to overwrite the PEP8 checks, add a `# noqa: CODE` at the end of the line (e.g. `# noqa: W605`).

Please fix any errors that are raised by these commands before submitting a pull request.

## Further Development

1. Train SHEATH with NRT SDOML (Embeddings, AR&CH Features), using OMNI as targets.

## Acknowledgements
This work is the research product of FDL-X Heliolab a public/private partnership between NASA, Trillium Technologies Inc (trillium.tech) and commercial AI partners Google Cloud, NVIDIA and Pasteur Labs & ISI, developing open science for all Humankind. 
