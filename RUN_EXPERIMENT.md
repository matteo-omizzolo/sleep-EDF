# Running the Experiment for Your Presentation

## Quick Start (5 minutes to results!)

### 1. Install Dependencies

```bash
cd "/Users/matteoomizzolo/Desktop/Github Projects/BayesianNonParametrics/sleep-EDF"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install numpy scipy scikit-learn matplotlib seaborn pyyaml
```

### 2. Run Complete Experiment

```bash
# Quick mode (~5 minutes) - for testing
python scripts/run_complete_experiment.py \
    --n-subjects 10 \
    --n-epochs 600 \
    --output results/presentation \
    --quick

# Full mode (~20 minutes) - for presentation
python scripts/run_complete_experiment.py \
    --n-subjects 15 \
    --n-epochs 800 \
    --output results/presentation_full
```

### 3. Find Your Figures

All figures will be saved to `results/presentation/figures/`:

- `fig1_posterior_num_states.pdf` - HDP vs iDP state counts
- `fig2_state_sharing_heatmap.pdf` - Who uses which state
- `fig3_dwell_times.pdf` - Sticky prior effect
- `fig4_predictive_performance.pdf` - LOSO test results
- `fig5_label_agreement.pdf` - ARI, NMI, F1 scores  
- `fig6_stick_breaking_weights.pdf` - Beta distribution
- `fig8_hypnogram_reconstruction.pdf` - Example predictions

Plus `summary_table.txt` with all metrics!

## What the Experiment Does

1. **Generates synthetic sleep data** (5 true sleep stages, realistic transitions)
2. **Trains HDP-HMM** (sticky, with hierarchical sharing)
3. **Trains iDP-HMM** (independent per subject, no sharing)
4. **Runs LOSO cross-validation** (tests generalization)
5. **Generates all 9 figures** + summary table

## Expected Results

Based on the synthetic data, you should see:

- **HDP-HMM discovers ~5 states** (matches true sleep stages)
- **iDP-HMM uses ~30-50 states total** (fragmented across subjects)
- **Longer dwell times with sticky prior** (~120-180s vs ~60s)
- **Better generalization (ARI ~0.7-0.8)** for HDP vs iDP (~0.4-0.6)

## Customization

### Use Real Sleep-EDF Data

If you want to use real data instead of synthetic:

1. Download Sleep-EDF:
   ```bash
   python scripts/download_data.py --output data/raw --n-subjects 10
   ```

2. Modify `run_complete_experiment.py` to load real data (see TODO in code)

### Adjust Parameters

Edit the script or pass arguments:
- `--n-subjects` - Number of subjects (more = slower but more robust)
- `--n-epochs` - Length of each recording
- `--seed` - Random seed for reproducibility
- `--quick` - Fewer MCMC iterations (faster)

## Troubleshooting

### "Import error" when running

Make sure virtual environment is activated and dependencies installed:
```bash
source .venv/bin/activate
pip install numpy scipy scikit-learn matplotlib seaborn pyyaml
```

### Script runs but no plots appear

Plots are saved to files automatically. Check `results/presentation/figures/`

### "Singular matrix" errors

Reduce `n_features` or increase data variance in `generate_synthetic_sleep_data()`

### Want faster results?

Use `--quick` flag:
```bash
python scripts/run_complete_experiment.py --quick --n-subjects 5
```

## For Your 20-Minute Talk

I recommend:

1. **Run full experiment once** to generate all figures (~20 min)
2. **Use figures 1, 2, 3, 4, 5, 6, 8 + table** in your slides
3. **Skip figures 7 & 9** (require additional ablation studies)
4. **Prepare 1-2 slide per figure** with the "Why" from the original spec

### Slide Structure Suggestion

1. Title + Problem (1 slide)
2. Data & Methods (2 slides) - explain HDP-HMM briefly
3. **Figure 1**: Parsimony (1 slide) - "HDP needs fewer states"
4. **Figure 2**: Sharing (1 slide) - "Subjects reuse states"
5. **Figure 3**: Dwell times (1 slide) - "Sticky prior works"
6. **Figure 4-5**: Performance (2 slides) - "Better generalization & validity"
7. **Figure 6**: Mechanism (1 slide) - "How HDP concentrates mass"
8. **Figure 8**: Example (1 slide) - "Real predictions look good"
9. **Table**: Summary (1 slide) - "All metrics favor HDP"
10. Conclusion (1 slide)

= **~13 slides for 20 minutes** = perfect pacing!

---

**Ready to run?**

```bash
python scripts/run_complete_experiment.py --output results/my_presentation
```

Good luck with your presentation! 🎯
