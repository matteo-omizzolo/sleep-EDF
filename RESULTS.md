# Results Interpretation Guide

Complete guide to understanding experimental outputs, figures, and metrics.

## Summary Table Interpretation

### Example Output

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
  iDP-HMM NMI:  0.651

Per-Class F1 Scores:
             Wake    N1     N2     N3     REM
HDP-HMM:    0.720  0.280  0.640  0.510  0.550
iDP-HMM:    0.680  0.210  0.580  0.470  0.480
```

### Metric Explanations

#### Model Complexity

**Effective K vs Total K**:
- **Effective K**: States with >1% probability mass (meaningful states)
- **Total K**: All instantiated states (including negligible ones)
- **Why different**: Stick-breaking creates many states with tiny weights

**Expected Values**:
- HDP Effective K: 5-7 (close to 5 true stages)
- HDP Total K: 7-10 (some exploration)
- iDP Total K: 25-35 (5 stages × 5 subjects, no sharing)

**Interpretation**:
- ✅ **Good**: HDP discovers ~6 shared states, iDP discovers ~30 total
- ❌ **Bad**: HDP K=1 (collapsed) or K=K_max (saturated)
- ❌ **Bad**: iDP K similar to HDP (not discovering per-subject states)

#### State Persistence

**Median Dwell Time**: How long subjects stay in each state

**Formula**: `median(consecutive epochs in same state) × 30 seconds`

**Expected Values**:
- HDP: 120-300 seconds (2-5 minutes) - **realistic sleep stages**
- iDP: Similar if κ is same

**Physiological Context**:
- Sleep stages last 1-20 minutes in reality
- Wake: 5-60 minutes
- N1: 1-7 minutes (transitional)
- N2: 10-25 minutes
- N3: 20-40 minutes (deep sleep)
- REM: 5-30 minutes

**Interpretation**:
- ✅ **Good**: 120+ seconds (captures realistic persistence)
- ❌ **Bad**: <60 seconds (over-segmentation, micro-states)
- ⚠️ **Warning**: >600 seconds (under-segmentation, missing stages)

#### Predictive Performance

**Test Log-Likelihood**: $\log p(x_{\text{test}} | \text{train})$

**Higher is better**: Better generalization to new subjects

**Typical Range**: -1000 to -2000 (depends on data scale)

**Expected Relationship**: $\text{LL}_{\text{HDP}} > \text{LL}_{\text{iDP}}$

**Why HDP Better**:
- Hierarchical sharing → learns global sleep stage patterns
- Reduces overfitting to individual subjects
- Statistical strength pooling across subjects

**Interpretation**:
- ✅ **Good**: HDP outperforms iDP by 50-200 log units
- ⚠️ **Concerning**: Similar performance (hierarchical structure not helping)
- ❌ **Bad**: iDP better (bug or hyperparameter issue)

#### Clustering Quality

**ARI (Adjusted Rand Index)**:
- Range: [-1, 1] (0 = random, 1 = perfect)
- Measures agreement with true labels
- Adjusted for chance

**Interpretation Scale**:
- < 0.2: Poor clustering
- 0.2-0.4: Weak agreement
- 0.4-0.6: Moderate agreement ✅
- 0.6-0.8: Strong agreement
- > 0.8: Excellent (rare in unsupervised setting)

**NMI (Normalized Mutual Information)**:
- Range: [0, 1]
- Information-theoretic similarity
- Less sensitive to class imbalance than ARI

**Interpretation Scale**:
- < 0.3: Poor clustering
- 0.3-0.5: Moderate
- 0.5-0.7: Good ✅
- > 0.7: Excellent

**Expected Performance**:
- HDP ARI: 0.4-0.6 (moderate to strong)
- HDP NMI: 0.5-0.7 (good)
- HDP > iDP on both metrics

**Why Unsupervised is Hard**:
- No labeled data during training
- Must discover structure from data alone
- Class imbalance (Wake 70%, N1 3%)
- Individual variation in sleep patterns

#### Per-Class F1 Scores

**Ranking (Expected)**:
1. **Wake** (0.6-0.8): Highest prevalence, most distinctive features
2. **N2** (0.5-0.7): Most common sleep stage, clear patterns
3. **REM** (0.4-0.6): Distinctive theta activity
4. **N3** (0.4-0.5): Clear delta dominance
5. **N1** (0.2-0.4): Hardest - transitional, only 3% of data

**Why N1 is Hardest**:
- Transitional state between wake and sleep
- Subtle features (slight theta increase)
- Only 3% of data (severe class imbalance)
- High inter-rater variability even among experts

**Interpretation**:
- ✅ **Good**: Ranking matches expected (Wake > ... > N1)
- ✅ **Good**: HDP better than iDP on most classes
- ⚠️ **Concerning**: F1 < 0.3 for all classes (check hyperparameters)
- ❌ **Bad**: Reversed ranking (e.g., N1 > Wake)

## Figure Interpretations

### Figure 1: Posterior over Number of States

**What it shows**: Distribution of K across MCMC iterations

**Key Observations**:
- **HDP**: Concentrated on 5-7 states (efficient discovery)
- **iDP**: Spread over 25-35 states (proliferation)

**Expected Pattern**:
```
HDP:  ▁▃█████▃▁  (peaked at K=6)
iDP:  ▁▁▃▅▇████▇▅▃▁  (spread across K=25-35)
```

**Interpretation**:
- ✅ **Good**: HDP concentrated, iDP spread out
- ⚠️ **Warning**: HDP at K=1 or K=K_max (model issue)
- ⚠️ **Warning**: iDP < 15 states (not discovering per-subject states)

### Figure 2: State Sharing Heatmap

**What it shows**: Which states each subject uses (HDP only)

**Axes**:
- X-axis: Global states (1 to K)
- Y-axis: Subjects (1 to M)
- Color: Usage probability (0 = never, 1 = always)

**Expected Pattern**:
- All subjects use states 1-5 (shared sleep stages)
- Sparse usage of states 6-12 (subject-specific variation)

**Interpretation**:
- ✅ **Good**: Clear shared structure (dark columns across all subjects)
- ✅ **Good**: 5-7 heavily used states
- ❌ **Bad**: No sharing (each subject uses different states)
- ❌ **Bad**: All subjects use all states uniformly (no structure)

### Figure 3: Dwell Time Distributions

**What it shows**: Duration of consecutive epochs in same state

**X-axis**: Dwell time in seconds (log scale)
**Y-axis**: Frequency/density

**Expected Pattern**:
- HDP: Peaked at 120-300 seconds (realistic)
- iDP: Similar if κ is same
- Both: Long tail (some states persist longer)

**Interpretation**:
- ✅ **Good**: Peak at 120+ seconds
- ❌ **Bad**: Peak at 30 seconds (over-segmentation)
- ⚠️ **Warning**: Uniform distribution (no persistence)

### Figure 4: Predictive Performance

**What it shows**: Test log-likelihood comparison (boxplot)

**Expected Pattern**:
```
      HDP        iDP
       |          |
   ----●----      |
       |       ---●---
       |          |
    higher      lower
```

**Interpretation**:
- ✅ **Good**: HDP median higher than iDP
- ✅ **Good**: HDP less variance (more stable)
- ❌ **Bad**: Overlapping distributions (no clear winner)

### Figure 5: Label Agreement

**What it shows**: ARI, NMI, Macro-F1 bar chart

**Expected Pattern**:
```
ARI:   [HDP: 0.54] [iDP: 0.40]
NMI:   [HDP: 0.65] [iDP: 0.45]
F1:    [HDP: 0.52] [iDP: 0.42]
```

**Interpretation**:
- ✅ **Good**: HDP bars consistently higher
- ✅ **Good**: 20-30% relative improvement
- ⚠️ **Concerning**: Small differences (<10%)

### Figure 6: Stick-Breaking Weights

**What it shows**: Global β distribution (HDP only)

**Left panel**: Bar plot with error bars
**Right panel**: Cumulative mass

**Expected Pattern**:
- Exponential decay: β₁ > β₂ > β₃ > ...
- 95% mass in first 5-7 states

**Interpretation**:
- ✅ **Good**: Clear exponential decay
- ✅ **Good**: 95% line crossed at K≈5-7
- ❌ **Bad**: Uniform weights (no structure)
- ❌ **Bad**: All mass on β₁ (collapsed)

### Figure 6b: Convergence Diagnostics

**What it shows**: MCMC convergence over iterations

**Panels**:
1. K trace (number of states vs iteration)
2. Beta evolution (top 5 weights over time)

**Expected Pattern**:
- K trace: Stabilizes after burn-in
- Beta weights: Converge to steady values

**Interpretation**:
- ✅ **Good**: Stable after burn-in (iteration 200)
- ⚠️ **Warning**: Still drifting at end (need more iterations)
- ❌ **Bad**: Oscillating wildly (poor mixing)

### Figure 8: Hypnogram Reconstruction

**What it shows**: Sleep staging comparison (first 300 epochs ≈ 2.5 hours)

**Three panels**:
1. True labels (ground truth)
2. HDP predictions (aligned)
3. iDP predictions (aligned)

**Expected Pattern**:
- HDP closer to true labels
- Captures major sleep cycles
- Some misalignment in transitions

**Interpretation**:
- ✅ **Good**: HDP tracks major patterns (N2 → N3 → REM cycles)
- ✅ **Good**: Captures wake periods correctly
- ⚠️ **Expected**: Misses some N1 (too brief/subtle)
- ❌ **Bad**: Random or uniform predictions

## Common Patterns

### Success Pattern

```
✅ HDP discovers 5-7 shared states
✅ iDP discovers 25-35 total states (no sharing)
✅ HDP dwell times: 120-300 seconds
✅ HDP > iDP in ARI, NMI, F1
✅ HDP > iDP in test log-likelihood
✅ Per-class ranking: Wake > N2 > REM > N3 > N1
✅ State sharing heatmap shows clear structure
✅ Stick-breaking weights decay exponentially
```

### Warning Pattern

```
⚠️ HDP K very close to iDP K (not much sharing)
⚠️ Small performance differences (< 10%)
⚠️ High variance in metrics across subjects
⚠️ K still drifting at end of MCMC
⚠️ Very low F1 for minority classes (<0.2)
```

**Actions**: Increase iterations, tune hyperparameters, check data quality

### Failure Pattern

```
❌ HDP K=1 (collapsed model)
❌ HDP K=K_max (saturated model)
❌ iDP outperforms HDP
❌ Dwell times < 60 seconds (over-segmentation)
❌ ARI < 0.3 (poor clustering)
❌ Reversed class ranking (N1 > Wake)
```

**Actions**: Debug model, check implementation, verify data

## Expected Performance Ranges

### By Number of Subjects

| Subjects | HDP K | iDP K | HDP ARI | Runtime |
|----------|-------|-------|---------|---------|
| 2 | 5-6 | 10-12 | 0.45-0.55 | ~5 min |
| 5 | 5-7 | 25-35 | 0.50-0.60 | ~15 min |
| 10 | 6-8 | 50-70 | 0.52-0.62 | ~30 min |
| 20 | 6-9 | 100-140 | 0.54-0.64 | ~60 min |

### By Hyperparameter Settings

| Setting | K | Dwell | ARI | Note |
|---------|---|-------|-----|------|
| γ=1, κ=50 | 3-4 | 180s | 0.40 | Too few states |
| γ=5, κ=50 | 5-7 | 180s | 0.55 | Balanced ✅ |
| γ=10, κ=50 | 8-12 | 180s | 0.50 | Proliferation |
| γ=5, κ=10 | 5-7 | 60s | 0.45 | Over-segmentation |
| γ=5, κ=100 | 2-3 | 360s | 0.35 | Under-segmentation |

## Comparison to Literature

### Sleep Staging Performance

| Method | Type | ARI | Macro-F1 | Note |
|--------|------|-----|----------|------|
| HDP-HMM (ours) | Unsupervised | 0.50-0.60 | 0.50-0.60 | No labels used |
| K-means | Unsupervised | 0.30-0.40 | 0.40-0.50 | Baseline |
| GMM-HMM | Unsupervised | 0.40-0.50 | 0.45-0.55 | Fixed K |
| Supervised CNN | Supervised | 0.75-0.85 | 0.75-0.85 | Uses labels |
| Human Expert | Supervised | 0.80-0.90 | 0.82-0.92 | Gold standard |

**Context**: Unsupervised methods typically achieve 60-70% of supervised performance.

### Hierarchical Model Benefits

**Typical Improvements** (HDP vs iDP):
- State discovery: 5× more efficient (6 vs 30 states)
- Predictive LL: +50-200 units
- ARI: +0.10-0.20 absolute
- F1: +0.05-0.15 absolute

---

**Last Updated**: November 2025
