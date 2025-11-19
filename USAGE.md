# Usage Guide

Complete guide to running experiments, interpreting outputs, and troubleshooting.

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd sleep-EDF

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

###2. Verify Installation

```bash
# Test imports
python -c "from src.models.simple_hdp_hmm import SimpleStickyHDPHMM; print('✓ Models OK')"
python -c "from src.data.load_sleep_edf import load_sleep_edf_dataset; print('✓ Data OK')"
python -c "from src.eval.metrics import compute_ari; print('✓ Eval OK')"
```

### 3. Run Quick Test

```bash
# Test with synthetic data (2 minutes)
python scripts/run_complete_experiment.py \
    --n-subjects 2 \
    --n-epochs 200 \
    --output results/test \
    --quick

# Check output
ls results/test/figures/  # Should have 7-8 PDF files
cat results/test/summary_table.txt
```

## Command-Line Interface

### Main Script: `run_complete_experiment.py`

```bash
python scripts/run_complete_experiment.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-real-data` | flag | False | Use Sleep-EDF dataset (vs synthetic) |
| `--n-subjects` | int | 10 | Number of subjects to use |
| `--n-epochs` | int | 800 | Epochs per subject (synthetic only) |
| `--output` | path | `results/output` | Output directory for results |
| `--quick` | flag | False | Quick mode: 200 iterations (vs 500) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--no-incremental-cache` | flag | False | Disable per-subject caching |

### Example Commands

#### 1. Full Experiment (Publication Quality)

```bash
# 5 subjects, 500 iterations, ~15 minutes
python scripts/run_complete_experiment.py \
    --use-real-data \
    --n-subjects 5 \
    --output results/publication \
    --seed 42
```

#### 2. Quick Development Test

```bash
# 5 subjects, 200 iterations, ~6 minutes
python scripts/run_complete_experiment.py \
    --use-real-data \
    --n-subjects 5 \
    --output results/dev_test \
    --quick
```

#### 3. Large-Scale Experiment

```bash
# 20 subjects, 500 iterations, ~60 minutes
python scripts/run_complete_experiment.py \
    --use-real-data \
    --n-subjects 20 \
    --output results/full_scale
```

#### 4. Synthetic Data Test

```bash
# 10 synthetic subjects for debugging
python scripts/run_complete_experiment.py \
    --n-subjects 10 \
    --n-epochs 800 \
    --output results/synthetic
```

## Experiment Workflow

### Pipeline Steps

The script executes these steps automatically:

```
[1/5] Loading data...
  - Load/cache Sleep-EDF recordings
  - Extract 72 features per epoch
  - Standardize features globally

[2/5] Training models...
  - Fit HDP-HMM (hierarchical)
  - Fit iDP-HMM (independent)
  - Print progress every 20 iterations

[3/5] Computing metrics...
  - Generate predictions for all subjects
  - Hungarian alignment to true labels
  - Compute ARI, NMI, F1 scores

[4/5] Generating figures...
  - 8 publication-quality PDFs
  - Saved to results/<output>/figures/

[5/5] Creating summary table...
  - LaTeX-ready comparison table
  - Saved to results/<output>/summary_table.txt
```

### Progress Monitoring

**Live Output**:
```
================================================================================
HDP-HMM FOR SLEEP STAGING - COMPLETE EXPERIMENT
================================================================================
Subjects: 5
Output: results/my_experiment
Data source: Real Sleep-EDF
Incremental cache: ON
================================================================================

[1/5] Loading data...
  Loading SC4001E0 from cache...
  Loading SC4002E0 from cache...
  ... (continues)
  Loaded 5 subjects, 12082 total epochs

[2/5] Training models...
  Training HDP-HMM...
Fitting Sticky HDP-HMM...
  Iteration 20/500, K=6
  Iteration 40/500, K=7
  ...
  Complete! Mean K=6.2

  Training independent DP-HMMs...
  Subject 1/5 ... Subject 5/5

[3/5] Computing metrics...
  Subject 1/5: HDP ARI=0.542, iDP ARI=0.401
  ... (continues)
```

**Log to File**:
```bash
# Save output to log file
python scripts/run_complete_experiment.py \
    --use-real-data \
    --output results/exp1 \
    2>&1 | tee experiment.log
```

## Output Structure

### Directory Layout

```
results/<output-name>/
├── figures/                               # Publication-ready PDFs
│   ├── fig1_posterior_num_states.pdf
│   ├── fig2_state_sharing_heatmap.pdf
│   ├── fig3_dwell_times.pdf
│   ├── fig4_predictive_performance.pdf
│   ├── fig5_label_agreement.pdf
│   ├── fig6_stick_breaking_weights.pdf
│   ├── fig6b_convergence_diagnostics.pdf
│   └── fig8_hypnogram_reconstruction.pdf
└── summary_table.txt                      # Quantitative results
```

### Summary Table Format

```
================================================================================
HDP-HMM vs iDP-HMM COMPARISON
================================================================================

Model Complexity:
  HDP-HMM Effective K:     6.2 (states with >1% mass)
  HDP-HMM Total K:        7.0 (all instantiated states)
  iDP-HMM Total K:       28.0 (sum across subjects)

State Persistence:
  HDP-HMM Median Dwell:   180.0 seconds (6.0 epochs)
  iDP-HMM Median Dwell:    30.0 seconds (1.0 epochs)

Predictive Performance:
  HDP-HMM Test Log-Likelihood:  -1245.3
  iDP-HMM Test Log-Likelihood:  -1389.7

Clustering Quality:
  HDP-HMM ARI:  0.542
  iDP-HMM ARI:  0.401
  
  HDP-HMM NMI:  0.651
  iDP-HMM NMI:  0.445

  HDP-HMM Macro-F1:  0.520
  iDP-HMM Macro-F1:  0.420

Per-Class F1 Scores:
             Wake    N1     N2     N3     REM
HDP-HMM:    0.720  0.280  0.640  0.510  0.550
iDP-HMM:    0.680  0.210  0.580  0.470  0.480
================================================================================
```

## Interpreting Results

### Key Metrics

**1. Number of States (K)**
- **HDP Effective K**: States with >1% probability mass
- **HDP Total K**: All instantiated states
- **iDP Total K**: Sum across all subjects
- **Expected**: HDP K ≈ 5-7, iDP K ≈ 25-35

**2. State Persistence**
- **Median Dwell Time**: Consecutive epochs in same state
- **Expected**: 180+ seconds (realistic sleep stage durations)
- **Too low** (<60s): Over-segmentation, need higher κ

**3. Predictive Performance**
- **Test Log-Likelihood**: Higher = better generalization
- **Expected**: HDP > iDP (hierarchical sharing helps)

**4. Clustering Quality**
- **ARI**: 0.4-0.6 = moderate, >0.6 = strong agreement
- **NMI**: 0.6-0.7 = good information preservation
- **Macro-F1**: 0.5-0.6 = balanced performance across classes

**5. Per-Class F1**
- **Wake**: Highest (most prevalent, distinctive)
- **N1**: Lowest (only 3% of data, transitional)
- **N2, N3, REM**: Intermediate (0.4-0.6)

### Success Criteria

**Experiment is successful if**:
- ✅ HDP discovers 5-8 states (not 1, not 12)
- ✅ HDP has longer dwell times than iDP
- ✅ HDP has higher ARI/NMI than iDP
- ✅ Per-class F1 follows expected ranking (Wake > REM > N2 > N3 > N1)

**Warning signs**:
- ❌ K=1 or K=K_max: Model collapsed or saturated
- ❌ Dwell time < 30s: Over-segmentation
- ❌ ARI < 0.3: Poor clustering (check data/hyperparameters)
- ❌ iDP outperforms HDP: Hierarchical structure not helping (investigate why)

## Troubleshooting

### Common Issues

#### 1. Model Collapses to K=1

**Symptoms**:
```
Iteration 100/500, K=1
Iteration 200/500, K=1
Complete! Mean K=1.0
```

**Causes**:
- γ too low (not enough state creation)
- κ too high (excessive self-transition bias)
- α too low (rigid transitions)

**Solutions**:
```python
# Increase state creation
gamma = 5.0  # was 1.0

# Reduce stickiness
kappa = 50.0  # was 100.0

# Increase flexibility
alpha = 10.0  # was 2.0
```

#### 2. Model Saturates at K=K_max

**Symptoms**:
```
Iteration 100/500, K=12
Iteration 200/500, K=12
Complete! Mean K=12.0
```

**Causes**:
- K_max too low for data complexity
- γ too high (excessive state proliferation)

**Solutions**:
```python
# Increase capacity
K_max = 20  # was 12

# Or reduce proliferation
gamma = 3.0  # was 5.0
```

#### 3. Poor Convergence (K Fluctuating)

**Symptoms**:
```
Iteration 100/500, K=5
Iteration 120/500, K=8
Iteration 140/500, K=6
... (continues fluctuating)
```

**Causes**:
- Not enough iterations
- Burn-in too short
- Poor initialization

**Solutions**:
```python
# Longer run
n_iter = 1000  # was 500
burn_in = 400  # was 200

# Or use full mode (not --quick)
```

#### 4. Memory Errors

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Causes**:
- Too many subjects
- Too many iterations stored

**Solutions**:
```bash
# Reduce subjects
--n-subjects 10  # was 20

# Or increase sample thinning (in code)
thin = 20  # Store every 20th sample
```

#### 5. Slow Data Loading

**Symptoms**:
- First run takes 45+ minutes on "Loading data..."

**Solutions**:
```bash
# Use incremental caching (default)
--use-real-data  # Caches automatically

# Force cache rebuild if corrupted
rm -rf data/processed/sleep_edf_subjects/
```

#### 6. NaN or Inf Values

**Symptoms**:
```
RuntimeWarning: invalid value encountered in log
ValueError: Input contains NaN
```

**Causes**:
- Numerical underflow in forward-backward
- Singular covariance matrices

**Solutions**:
- Already handled in code with fallbacks
- If persistent, check data quality
- Ensure features are standardized

### Validation Checks

Run these commands to verify installation:

```bash
# Check Python version (need 3.9+)
python --version

# Check imports
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# Check data directory
ls data/raw/sleep-cassette/ | head -5  # Should show .edf files

# Check cache
ls data/processed/sleep_edf_subjects/ | head -5  # Should show .npz files

# Test feature extraction
python -c "
from src.data.load_sleep_edf import load_sleep_edf_dataset
from pathlib import Path
X, y = load_sleep_edf_dataset(
    Path('data/raw/sleep-cassette'),
    n_subjects=1,
    verbose=False
)
assert X[0].shape[1] == 72, 'Wrong feature dimension'
print(f'✓ Features: {X[0].shape}')
"
```

## Advanced Usage

### Custom Hyperparameters

Edit `scripts/run_complete_experiment.py` to modify model hyperparameters:

```python
hdp_model = SimpleStickyHDPHMM(
    K_max=15,        # Increase for more capacity
    gamma=10.0,      # Increase for more states
    alpha=5.0,       # Decrease for rigid transitions
    kappa=100.0,     # Increase for longer dwell times
    n_iter=1000,     # Increase for better convergence
    burn_in=400,
    random_state=42,
    verbose=1
)
```

### Running Multiple Experiments

```bash
# Parameter sweep
for kappa in 10 50 100; do
    python scripts/run_complete_experiment.py \
        --use-real-data \
        --output results/kappa_${kappa} \
        --quick
done

# Compare results
grep "HDP-HMM ARI" results/kappa_*/summary_table.txt
```

### Plotting Individual Figures

```python
from src.eval.plots import plot_posterior_num_states
from src.models.simple_hdp_hmm import SimpleStickyHDPHMM

# Load model (assumes you saved it)
model = SimpleStickyHDPHMM(...)
model.fit(X_list)

# Generate single figure
plot_posterior_num_states(
    hdp_samples=model.samples_,
    idp_samples=None,  # Optional
    output_path='my_figure.pdf'
)
```

## Performance Tips

### Speed Optimization

1. **Use caching** (automatic with `--use-real-data`)
2. **Quick mode** for development (`--quick` → 200 iterations)
3. **Fewer subjects** during debugging (`--n-subjects 2`)
4. **Reduce figure generation** (comment out in script)

### Memory Optimization

1. **Store fewer samples**: Increase `thin` parameter
2. **Reduce subjects**: Use `--n-subjects 10` instead of 20
3. **Clear old results**: `rm -rf results/old_*`

---

**Last Updated**: November 2025
