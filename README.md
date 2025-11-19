# Hierarchical Dirichlet Process Hidden Markov Model for Sleep Stage Classification

A Bayesian nonparametric approach demonstrating the **superiority of hierarchical models over independent models** in grouped time-series data.

## �� Research Objective

**Central Hypothesis**: Hierarchical Bayesian models outperform independent models for grouped data where subjects share common underlying patterns.

**Test Domain**: Sleep stage classification using polysomnography data
- **Why**: All humans share the same 5 sleep stages (W, N1, N2, N3, REM)
- **Challenge**: Individual variation in stage durations and transitions
- **Ideal testbed**: Universal structure + individual variation

## 🔬 Scientific Contributions

### Primary Contributions

1. **Hierarchical vs Independent Comparison**
   - HDP-HMM: Shares global state distribution across subjects
   - iDP-HMM: Fits independent models per subject
   - Quantifies value of hierarchical sharing

2. **State Discovery**
   - HDP discovers 5-7 shared states (efficient)
   - iDP discovers 25-35 total states (inefficient)
   - Demonstrates parameter efficiency: 5× reduction

3. **Generalization Performance**
   - HDP shows better out-of-sample prediction
   - Higher ARI/NMI scores (better clustering)
   - Superior handling of class imbalance

### Key Results

| Metric | HDP-HMM | iDP-HMM |
|--------|---------|---------|
| **States Discovered** | 5-7 shared | 25-35 total |
| **Test Log-Likelihood** | Higher | Lower |
| **ARI Score** | ~0.55 | ~0.40 |
| **Parameter Efficiency** | O(K) | O(K×M) |

## 📁 Repository Structure

```
sleep-EDF/
├── README.md                   # This file
├── METHODOLOGY.md              # Mathematical formulation
├── IMPLEMENTATION.md            # Code architecture
├── USAGE.md                    # How to run experiments
├── RESULTS.md                  # Interpreting outputs
├── requirements.txt            # Dependencies
│
├── data/
│   ├── raw/sleep-cassette/    # Sleep-EDF dataset (20 subjects)
│   └── processed/             # Cached features (.npz)
│
├── src/
│   ├── data/                  # Data loading & preprocessing
│   │   └── load_sleep_edf.py
│   ├── models/                # Core implementations
│   │   └── simple_hdp_hmm.py # Sticky HDP-HMM
│   └── eval/                  # Metrics & visualization
│       ├── metrics.py
│       ├── hungarian.py
│       └── plots.py
│
├── scripts/
│   └── run_complete_experiment.py  # Main pipeline
│
└── results/
    ├── improved/               # Latest results
    └── full_experiment/        # Full-scale experiment
```

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd sleep-EDF
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Experiment

```bash
# Full experiment (5 subjects, 500 iterations, ~15 minutes)
python scripts/run_complete_experiment.py \
    --n-subjects 5 \
    --output results/my_experiment \
    --use-real-data

# Quick test (200 iterations, ~6 minutes)
python scripts/run_complete_experiment.py \
    --n-subjects 5 \
    --output results/quick_test \
    --use-real-data \
    --quick
```

### View Results

Results saved to `results/<output-name>/`:
- `figures/*.pdf` - 8 publication-quality visualizations
- `summary_table.txt` - Quantitative comparison

## 📊 Key Figures

1. **Posterior over K**: Shows HDP concentrates on few states, iDP proliferates
2. **State Sharing Heatmap**: Cross-subject sharing patterns
3. **Dwell Times**: Both models capture realistic stage persistence
4. **Predictive Performance**: HDP superior generalization
5. **Label Agreement**: HDP better aligns with ground truth
6. **Stick-Breaking Weights**: Mass concentration on shared states
7. **Convergence Diagnostics**: MCMC convergence verification
8. **Hypnogram Reconstruction**: Example sleep staging

## 📖 Documentation

- **[METHODOLOGY.md](METHODOLOGY.md)**: Mathematical formulation, inference algorithms, hyperparameter rationale
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: Code architecture, design decisions, optimization details
- **[USAGE.md](USAGE.md)**: Detailed usage guide, command-line options, troubleshooting
- **[RESULTS.md](RESULTS.md)**: How to interpret outputs, expected performance, figure descriptions

## 🔑 Key Technical Details

### Model Hyperparameters

```python
HDP-HMM Configuration:
  K_max = 12      # Capacity for state discovery
  γ = 5.0         # Global DP concentration (allows 5-8 states)
  α = 10.0        # Transition flexibility
  κ = 50.0        # Sticky self-transition bias
  n_iter = 500    # MCMC iterations
  burn_in = 200   # Burn-in period
```

**Rationale**:
- γ=5.0: Expected K ≈ 5-8 (matches 5 true sleep stages)
- α=10.0: Flexible transitions between stages
- κ=50.0: Self-transition bias = 0.83 (realistic persistence)

### Dataset: Sleep-EDF

- **Source**: PhysioNet Sleep-EDF Database
- **Subjects**: 20 healthy adults
- **Features**: 72-dimensional per 30-second epoch
  - Spectral band powers (δ, θ, α, β, γ)
  - Spectral ratios (θ/α, α/δ)
  - Hjorth parameters (variance, mobility)
- **Labels**: Expert-annotated 5 sleep stages
- **Class Distribution**: Wake 70%, N2 16%, N3 6%, REM 6%, N1 3%

## 📚 References

### Theoretical Foundation

1. **Teh et al. (2006)**. "Hierarchical Dirichlet Processes." JASA.
2. **Fox et al. (2008)**. "A Sticky HDP-HMM for Systems with State Persistence." ICML.

### Dataset

3. **Kemp et al. (2000)**. "Analysis of a Sleep-Dependent Neuronal Feedback Loop." IEEE-BME.
4. **PhysioNet**: https://physionet.org/content/sleep-edfx/

## 📜 Citation

```bibtex
@software{hdp_hmm_sleep_2025,
  title = {Hierarchical Dirichlet Process HMM for Sleep Stage Classification},
  author = {[Your Name]},
  year = {2025},
  url = {[Repository URL]}
}
```

## 📄 License

MIT License - see LICENSE file

## 🤝 Contact

For questions or issues, please open a GitHub issue.

---

**Status**: Production-ready | Last Updated: November 2025
