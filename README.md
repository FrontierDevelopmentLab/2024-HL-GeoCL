# 2024-HL-GeoCL
## FDL-X Heliolab 2024: Geoeffectiveness Continuous Learning

Currently two modules are included in the geocloak folder, namely DAGGER and SHEATH.

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
uv venv --python 3.10
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
