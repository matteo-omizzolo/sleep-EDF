# Model Improvements for Better Sleep Staging Performance

## Summary of Changes

This document outlines the improvements made to align the model performance with state-of-the-art sleep staging results reported in the literature.

## 1. Enhanced Feature Extraction

### Previous Features (10 total):
- 5 spectral bands × 2 channels = 10 features
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)  
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-50 Hz)

### New Features (18 total):
- **Spectral band powers**: 5 bands × 2 channels = 10 features
- **Spectral ratios**: 2 ratios × 2 channels = 4 features
  - Theta/Alpha ratio (important for drowsiness detection)
  - Alpha/Delta ratio (distinguishes wake from sleep)
- **Temporal features (Hjorth parameters)**: 2 features × 2 channels = 4 features
  - Variance (activity level)
  - Mobility (mean frequency estimate)

**Rationale**: Sleep staging literature shows that spectral ratios and temporal features significantly improve discrimination between similar stages (especially N1 vs Wake, N2 vs N3).

## 2. Improved Hyperparameters

### Previous Values:
```python
gamma = 1.0   # Global concentration
alpha = 1.0   # DP concentration  
kappa = 10.0  # Stickiness parameter
```

### New Values:
```python
gamma = 2.0   # Increased for better state discovery
alpha = 5.0   # Increased for more flexible transitions
kappa = 50.0  # Significantly increased for realistic sleep stage durations
```

**Rationale**: 
- **γ = 2.0**: Allows model to discover more components initially, then prune unused ones
- **α = 5.0**: Permits richer transition structure while maintaining hierarchy
- **κ = 50.0**: Sleep stages typically last 2-20 minutes (4-40 epochs), requiring strong self-transition bias

## 3. Better Initialization Strategy

### Previous Approach:
- K-means clustering with 10 clusters
- Uniform beta weights

### New Approach:
- **Hierarchical (Ward) clustering** with 12 initial clusters
  - Better captures nested structure of sleep stages
  - More robust to outliers than K-means
- **Informed beta weights**: Concentrates mass on active clusters
- **Feature standardization** before clustering

**Rationale**: Ward linkage is more appropriate for Gaussian emissions and captures the hierarchical nature of sleep stages (N3 → N2 → N1 → W progression).

## 4. Duration-Aware State Modeling

### Previous:
- Simple Markov transitions (memoryless)

### New:
- **Duration means initialization**: Gamma(5, 3) prior (mean ≈ 15 epochs = 7.5 minutes)
- **Duration-aware sampling**: Combines posterior with squared transitions for stronger persistence
- **Duration parameters tracked**: Enables duration-based state analysis

**Rationale**: Sleep stages have characteristic durations (e.g., N3 bouts typically last 10-30 minutes). True HSMM modeling requires explicit duration distributions.

## 5. Feature Standardization

### New Addition:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
```

**Rationale**: Different feature types (power, ratios, variance) have different scales. Standardization ensures all features contribute equally to the model, preventing high-variance features from dominating.

## 6. Increased Training Iterations

### Previous:
- Quick mode: 150 iterations
- Full mode: 300 iterations

### New:
- Quick mode: 200 iterations  
- Full mode: 500 iterations
- Burn-in: 200 iterations (40% of full run)

**Rationale**: The complex posterior with 15-20 latent states requires more iterations for proper mixing, especially with the stronger stickiness parameter.

## Expected Performance Improvements

Based on these changes, we expect:

### Clustering Metrics:
- **ARI**: 0.35 → 0.50-0.60 (better agreement with true sleep stages)
- **NMI**: 0.45 → 0.60-0.70 (better mutual information)
- **Macro-F1**: 0.15 → 0.45-0.60 (much better classification after alignment)

### Model Properties:
- **Number of states**: More consistent (~8-12 states vs wide variation)
- **State persistence**: Longer, more realistic dwell times (5-20 minutes)
- **Per-class F1**: 
  - Wake: 0.6-0.8 (easiest to detect)
  - N2: 0.5-0.7 (most common stage)
  - REM: 0.4-0.6 (distinctive spectral features)
  - N3: 0.4-0.5 (deep sleep, clear delta dominance)
  - N1: 0.2-0.4 (hardest: transitional stage, only 3% of data)

## References

These improvements are based on:

1. **Fox et al. (2011)**: "A Sticky HDP-HMM with Application to Speaker Diarization"
   - Optimal kappa values: 50-100 for strong persistence
   
2. **Chriskos et al. (2020)**: "A review on current trends in automatic sleep staging through bio-signal recordings"
   - Importance of spectral ratios and Hjorth parameters

3. **Chambon et al. (2018)**: "A Deep Learning Architecture for Temporal Sleep Stage Classification Using Multivariate and Multimodal Time Series"
   - Feature importance analysis for sleep staging

4. **Teh et al. (2006)**: "Hierarchical Dirichlet Processes"
   - Hyperparameter selection guidelines

## Running the Improved Model

```bash
# Quick test (2 subjects, 200 iterations)
python scripts/run_complete_experiment.py \
    --use-real-data \
    --n-subjects 2 \
    --output results/improved_test \
    --quick

# Full experiment (20 subjects, 500 iterations)
python scripts/run_complete_experiment.py \
    --use-real-data \
    --n-subjects 20 \
    --output results/improved_full
```

## Notes

- The improved features increase dimensionality from 10 to 18, which may slightly increase computation time (~20-30% longer)
- Feature standardization is critical - without it, the power features dominate and ratios/temporal features are ignored
- The stronger kappa (50.0) may initially appear to over-smooth, but this is appropriate for sleep staging where stages last minutes, not seconds
- Results may take 1-2 hours for full 20-subject experiment with 500 iterations

