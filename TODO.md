# Next Steps & Development Roadmap

## Immediate Next Steps (Before Running Experiments)

### 1. Set Up Python Environment

```bash
cd "/Users/matteoomizzolo/Desktop/Github Projects/BayesianNonParametrics/sleep-EDF"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Test installation
python tests/test_installation.py
```

### 2. Push to GitHub

Follow instructions in `GITHUB_SETUP.md` to create GitHub repository and push.

### 3. Download Data

```bash
python scripts/download_data.py --output data/raw --n-subjects 10
```

## Development Priorities

### High Priority (Core Functionality)

- [ ] **Complete MCMC inference implementation** in `src/models/hdp_hmm_sticky.py`
  - Full Gibbs updates for β (global stick-breaking weights)
  - Full Gibbs updates for π (transition matrices)
  - Full Gibbs updates for μ, Σ (emission parameters)
  - Hyperparameter updates (γ, α, κ) with Metropolis-Hastings

- [ ] **Implement Viterbi decoding** for state sequence prediction
  - Add `viterbi()` method to `StickyHDPHMM`
  - More efficient than forward-backward for prediction

- [ ] **Complete preprocessing pipeline** in `src/data/preprocessing.py`
  - Create `preprocess_all.py` script
  - Handle multiple subjects efficiently
  - Cache preprocessed features

- [ ] **Implement baseline models**
  - `src/models/idp_hmm.py` - Independent DP-HMM
  - `src/models/pooled_ihmm.py` - Pooled infinite HMM
  - `scripts/run_idp_hmm.py` - Script for baseline

### Medium Priority (Evaluation & Visualization)

- [ ] **LOSO cross-validation** (`scripts/run_loso_cv.py`)
  - Leave-one-subject-out evaluation
  - Proper train/test splitting
  - Hungarian alignment on training set

- [ ] **Visualization functions** in `src/eval/plots.py`
  - Posterior over number of states K
  - State-sharing heatmap
  - Dwell-time distributions
  - Hypnogram reconstructions
  - Transition matrices
  - Stick-breaking weights β
  - Trace plots for MCMC diagnostics

- [ ] **Results generation** (`scripts/generate_figures.py`)
  - Generate all figures for paper/talk
  - Create summary tables
  - Export to PDF/PNG

- [ ] **Jupyter notebooks** in `notebooks/`
  - Data exploration
  - Feature analysis
  - Model comparison
  - Results visualization

### Lower Priority (Polish & Extensions)

- [ ] **MCMC diagnostics** in `src/inference/diagnostics.py`
  - Gelman-Rubin R-hat
  - Effective sample size (ESS)
  - Trace plots
  - Autocorrelation plots

- [ ] **Unit tests** in `tests/`
  - Test data loading
  - Test feature extraction
  - Test model fitting
  - Test evaluation metrics

- [ ] **Documentation**
  - API documentation with Sphinx
  - Tutorial notebooks
  - Mathematical derivations

- [ ] **Performance optimization**
  - Numba JIT compilation for hot loops
  - Parallel MCMC chains
  - Efficient sparse matrix operations

## Optional Extensions (Research Directions)

- [ ] **HDP-HSMM** (Hidden Semi-Markov Model)
  - Explicit duration modeling
  - Better for sleep stage transitions

- [ ] **Structured sticky prior**
  - Domain knowledge about allowed transitions
  - W → N1 → N2 → N3 → REM structure

- [ ] **Multi-channel extensions**
  - Shared states across EEG/EOG/EMG channels
  - Channel-specific emissions

- [ ] **Online inference**
  - Sequential Monte Carlo / particle filters
  - Real-time sleep staging

- [ ] **Deep learning comparison**
  - Compare to LSTM/Transformer baselines
  - Hybrid models (neural emissions + HDP structure)

## File Structure TODO

Still need to create:

```
sleep-EDF/
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── sampler.py          # MCMC samplers
│   │   ├── weak_limit.py       # Truncated stick-breaking
│   │   └── diagnostics.py      # Convergence checks
│   │
│   ├── eval/
│   │   └── plots.py            # Visualization functions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Config loading
│       └── logger.py           # Logging setup
│
├── scripts/
│   ├── preprocess_all.py       # Preprocess all subjects
│   ├── run_idp_hmm.py          # Baseline model
│   ├── run_loso_cv.py          # Cross-validation
│   └── generate_figures.py     # Figure generation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_results_visualization.ipynb
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_inference.py
```

## Timeline Suggestion (for 20-minute talk preparation)

### Week 1: Core Implementation
- Complete MCMC inference
- Implement baseline models
- Test on small dataset (5 subjects)

### Week 2: Experiments
- Run LOSO cross-validation
- Generate all figures and tables
- Verify results match expectations

### Week 3: Polish & Presentation
- Create presentation slides
- Polish visualizations
- Prepare speaker notes
- Practice talk

## Resources & References

Key papers to review while implementing:
- Fox et al. (2011) for sticky HDP-HMM details
- Teh et al. (2006) for HDP theory
- Van Gael et al. (2008) for beam sampling

Python libraries for reference:
- `pyhsmm` - Original HDP-HMM implementation (unmaintained)
- `ssm` - Modern state-space models
- `arviz` - Bayesian inference diagnostics

## Questions to Address

1. How many MCMC iterations are needed for convergence?
2. What K_max (truncation level) is sufficient?
3. How sensitive are results to hyperpriors (γ, α, κ)?
4. How much does stickiness κ improve over standard HDP-HMM?
5. Do we get 5 states (matching sleep stages) or more/less?

## Success Metrics

By the end, you should have:
- ✓ Working sticky HDP-HMM implementation
- ✓ Baseline comparisons (iDP-HMM, pooled iHMM)
- ✓ 9 publication-quality figures
- ✓ 1 summary table
- ✓ Reproducible results (fixed seed)
- ✓ 20-minute presentation ready

---

**Current Status**: 🟢 Repository initialized, ready for implementation!

**Next Action**: Set up Python environment and test installation
