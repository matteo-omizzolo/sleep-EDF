# Sleep-EDF HDP-HMM Project

Quick start guide for the Hierarchical Dirichlet Process HMM for sleep staging project.

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py --output data/raw --n-subjects 30
```

### 2. Run Basic Experiment

```bash
python scripts/run_hdp_hmm_sticky.py \
    --data data/processed \
    --config configs/default_config.yaml \
    --output results/demo
```

### 3. Reproduce All Results

```bash
bash scripts/reproduce_all.sh
```

## Project Structure

- `src/` - Core implementation
  - `data/` - Data loading and preprocessing
  - `models/` - HMM model implementations
  - `eval/` - Evaluation metrics
- `scripts/` - Executable scripts
- `configs/` - Configuration files
- `notebooks/` - Jupyter notebooks for analysis

## Key Files

- `src/models/hdp_hmm_sticky.py` - Main sticky HDP-HMM implementation
- `src/data/download.py` - PhysioNet data downloader
- `configs/default_config.yaml` - Default hyperparameters

## For More Details

See the full [README.md](README.md) for complete documentation.
