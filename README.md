# Hierarchical Dirichlet Process Hidden Semi-Markov Model for Sleep Staging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Bayesian nonparametric approach to unsupervised sleep staging using the sticky HDP-HMM on PhysioNet Sleep-EDF data.

## Overview

This project implements and evaluates hierarchical Dirichlet process hidden Markov models (HDP-HMMs) with sticky self-transitions for unsupervised sleep stage segmentation. The work directly extends the theoretical framework of:

- **Teh et al. (2006)**: Hierarchical Dirichlet Processes for sharing mixture components across groups
- **Fox et al. (2011)**: Sticky HDP-HMM for realistic state persistence in time series

### Research Question

**In unsupervised sleep staging, does hierarchical sharing of latent states across subjects via a sticky HDP-HMM improve (i) predictive performance and (ii) parsimony/consistency of discovered states—relative to fitting independent nonparametric HMMs per subject?**

## Key Features

- **Hierarchical state sharing**: Global stick-breaking prior enables state reuse across subjects
- **Sticky self-transitions**: Parameter κ encourages realistic dwell times (avoids micro-segmentation)
- **Rigorous evaluation**: Leave-one-subject-out (LOSO) cross-validation with predictive likelihood and label agreement metrics
- **Novel domain**: First application of sticky HDP-HMM to polysomnography sleep staging

## Dataset: Sleep-EDF Expanded

- **Source**: [PhysioNet Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Content**: 197 whole-night polysomnography recordings with expert hypnogram labels
- **Signals**: EEG (Fpz-Cz, Pz-Oz), EOG, EMG at 100 Hz
- **Labels**: Sleep stages {W, N1, N2, N3, REM} in 30-second epochs

### Why This Dataset?

- **Groups**: Each subject (or night) represents a group in the HDP hierarchy
- **Sequences**: Time series of 30-second epochs naturally map to HMM observations
- **Shared structure**: Biological sleep stages recur across subjects → ideal for hierarchical sharing
- **Labeled ground truth**: Enables unsupervised-to-supervised evaluation via Hungarian matching

## Models

### 1. Independent DP-HMM (iDP-HMM) — Baseline

Each subject m has its own infinite HMM with DP prior over transitions:
- No cross-subject sharing of states or parameters
- Baseline to quantify the value of hierarchical structure

### 2. Sticky HDP-HMM — Target Model

- **Global stick-breaking**: β ~ GEM(γ) shared across all subjects
- **Subject-specific transitions**: π_j^(m) ~ DP(α+κ, (αβ + κδ_j)/(α+κ))
- **Stickiness parameter**: κ biases self-transitions to encourage realistic dwell times
- **Gaussian emissions**: y_t | s_t ~ N(μ_k, Σ_k) with NIW prior

### 3. Pooled iHMM (Optional Control)

Single infinite HMM fit to all subjects jointly (no group variation) to demonstrate over-merging when hierarchy is removed.

## Project Structure

```
sleep-EDF/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
│
├── configs/                     # Configuration files
│   ├── default_config.yaml      # Default hyperparameters
│   └── experiment_configs/      # Specific experiment settings
│
├── data/                        # Data directory (not tracked)
│   ├── raw/                     # Raw Sleep-EDF files (.edf, .txt)
│   └── processed/               # Preprocessed features (.npy, .pkl)
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── download.py          # PhysioNet data downloader
│   │   ├── loader.py            # EDF file reader
│   │   └── preprocessing.py     # Feature extraction (PSD, bandpowers)
│   │
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Base HMM interface
│   │   ├── idp_hmm.py           # Independent DP-HMM per subject
│   │   ├── hdp_hmm_sticky.py    # Sticky HDP-HMM (main model)
│   │   └── pooled_ihmm.py       # Single pooled infinite HMM
│   │
│   ├── inference/               # MCMC inference
│   │   ├── __init__.py
│   │   ├── sampler.py           # Gibbs/beam sampler
│   │   ├── weak_limit.py        # Truncated stick-breaking approximation
│   │   └── diagnostics.py       # Convergence checks (R-hat, ESS)
│   │
│   ├── eval/                    # Evaluation metrics and plotting
│   │   ├── __init__.py
│   │   ├── metrics.py           # ARI, NMI, F1, log-likelihood
│   │   ├── hungarian.py         # State-to-label alignment
│   │   └── plots.py             # All visualization functions
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration loading
│       └── logger.py            # Logging setup
│
├── scripts/                     # Executable scripts
│   ├── download_data.py         # Download Sleep-EDF from PhysioNet
│   ├── preprocess_all.py        # Preprocess all subjects
│   ├── run_idp_hmm.py           # Run independent baseline
│   ├── run_hdp_hmm_sticky.py    # Run sticky HDP-HMM
│   └── run_loso_cv.py           # Full LOSO cross-validation
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_results_visualization.ipynb
│
├── results/                     # Output directory (not tracked)
│   ├── figures/                 # Generated plots
│   ├── tables/                  # Summary tables (CSV/LaTeX)
│   └── models/                  # Saved posterior samples (.pkl)
│
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_data.py
    ├── test_models.py
    └── test_inference.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/matteo-omizzolo/sleep-EDF.git
cd sleep-EDF

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Usage

### 1. Download Data

```bash
python scripts/download_data.py --output data/raw --n-subjects 30
```

### 2. Preprocess Features

```bash
python scripts/preprocess_all.py \
    --input data/raw \
    --output data/processed \
    --features psd bandpower
```

Features extracted per 30-second epoch:
- Welch power spectral density (PSD)
- Log bandpowers: δ (0.5-4 Hz), θ (4-8 Hz), α (8-12 Hz), σ (12-16 Hz), β (16-30 Hz)
- EOG variance, EMG variance (optional)

### 3. Run Models

#### Independent DP-HMM (baseline)

```bash
python scripts/run_idp_hmm.py \
    --data data/processed \
    --config configs/default_config.yaml \
    --output results/idp_hmm
```

#### Sticky HDP-HMM (target)

```bash
python scripts/run_hdp_hmm_sticky.py \
    --data data/processed \
    --config configs/default_config.yaml \
    --output results/hdp_hmm_sticky
```

#### Full LOSO Cross-Validation

```bash
python scripts/run_loso_cv.py \
    --data data/processed \
    --models idp_hmm hdp_hmm_sticky \
    --n-folds 30 \
    --output results/loso_cv
```

### 4. Analyze Results

Launch Jupyter notebooks:

```bash
jupyter notebook notebooks/03_model_comparison.ipynb
```

Or generate all figures programmatically:

```bash
python scripts/generate_figures.py --input results/loso_cv --output results/figures
```

## Experimental Design

### Cross-Validation Strategy

- **Leave-One-Subject-Out (LOSO)**: Fit on M-1 subjects, evaluate on held-out subject
- **Splits**: 20-30 subjects, 1 night per subject
- **Metrics**:
  - Predictive log-likelihood (test sequence)
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Macro-F1 score (post Hungarian alignment)

### Hyperparameters & Priors

```yaml
# Default configuration (configs/default_config.yaml)
concentration:
  gamma: Gamma(1.0, 1.0)     # Global DP concentration
  alpha: Gamma(1.0, 1.0)     # Group-level DP concentration
  kappa: Gamma(5.0, 1.0)     # Sticky self-transition bias

emissions:
  prior: NIW                  # Normal-Inverse-Wishart
  mu_0: [0, ..., 0]          # Prior mean (feature dim)
  kappa_0: 0.01              # Prior precision scaling
  psi: I                     # Prior scale matrix
  nu: feature_dim + 2        # Prior degrees of freedom

inference:
  method: weak_limit          # or 'beam_sampler'
  K_max: 50                  # Truncation level
  n_iter: 5000               # Total MCMC iterations
  burn_in: 2000              # Burn-in iterations
  thin: 5                    # Thinning interval
  n_chains: 3                # Parallel chains for diagnostics
```

### MCMC Diagnostics

- **Convergence**: Gelman-Rubin R-hat < 1.1 for key parameters
- **Mixing**: Effective sample size (ESS) > 100
- **Trace plots**: Visual inspection of α, γ, κ, K
- **Label switching**: Addressed via sticky prior and post-processing

## Key Results (Expected)

### Figures & Tables for 20-Minute Talk

All figures are automatically generated and saved to `results/figures/`:

1. **`fig1_posterior_num_states.pdf`**: Posterior over K (global states vs per-subject fragmentation)
2. **`fig2_state_sharing_heatmap.pdf`**: Who uses which state (subjects × global states)
3. **`fig3_dwell_times.pdf`**: Sticky matters—realistic segment durations
4. **`fig4_test_loglik_loso.pdf`**: Generalization to new subjects (boxplot)
5. **`fig5_ari_nmi_comparison.pdf`**: Biological plausibility (label agreement)
6. **`fig6_states_vs_subjects.pdf`**: E[K] growth with M (data scaling)
7. **`fig7_stick_breaking_weights.pdf`**: Global β (posterior mean ± 95% CI)
8. **`fig8_hypnogram_examples.pdf`**: Representative reconstructions (2-3 subjects)
9. **`fig9_ablation_kappa.pdf`**: Effect of stickiness κ on ARI/log-likelihood

**Table 1** (`tables/summary_table.tex`):

| Model            | E[K] | Median Dwell (s) | Test Log-Lik | ARI   | NMI   | Macro-F1 |
|------------------|------|------------------|--------------|-------|-------|----------|
| iDP-HMM          | ...  | ...              | ...          | ...   | ...   | ...      |
| HDP-HMM (sticky) | ...  | ...              | ...          | ...   | ...   | ...      |
| Pooled iHMM      | ...  | ...              | ...          | ...   | ...   | ...      |

## Alignment with Original Papers

### Teh et al. (2006): Hierarchical Dirichlet Processes
- **Core contribution**: Chinese restaurant franchise for sharing mixture components across groups
- **Our application**: Subjects = groups; sleep stages = shared components

### Fox et al. (2011): Sticky HDP-HMM
- **Core contribution**: κ parameter biases self-transitions → realistic dwell times
- **Our application**: Corrects over-segmentation in sleep time series

### Novel contribution of this work
- **Different domain**: Sleep staging (polysomnography) vs text topics or speaker diarization
- **Cross-subject generalization**: LOSO evaluation stresses utility of hierarchical sharing under distribution shift

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds in all scripts
- Configuration files track all hyperparameters
- MCMC diagnostics saved alongside results
- Python environment pinned via `requirements.txt`

To reproduce results from the paper/talk:

```bash
bash scripts/reproduce_all.sh
```

This script:
1. Downloads data
2. Preprocesses features
3. Runs all models with LOSO CV
4. Generates all figures and tables
5. Outputs to `results/paper/`

## References

### Core Methods
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). *Hierarchical Dirichlet processes*. JASA.
- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). *A sticky HDP-HMM with application to speaker diarization*. Annals of Applied Statistics.
- Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008). *Beam sampling for the infinite hidden Markov model*. ICML.

### Dataset
- Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A., & Oberye, J. J. (2000). *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG*. IEEE-BME.
- Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet*. Circulation.

### Related Work
- Stephens, M. (2000). *Dealing with label switching in mixture models*. JRSS-B.
- Johnson, M. J. & Willsky, A. S. (2013). *Bayesian nonparametric hidden semi-Markov models*. JMLR.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{omizzolo2025hdphsmm_sleep,
  author = {Omizzolo, Matteo},
  title = {Hierarchical Dirichlet Process Hidden Semi-Markov Model for Sleep Staging},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/matteo-omizzolo/sleep-EDF}
}
```

## Contact

Matteo Omizzolo - [GitHub](https://github.com/matteo-omizzolo)

---

**Status**: 🚧 Work in progress — initial implementation phase
