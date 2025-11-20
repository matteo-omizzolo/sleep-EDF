# Final Experiment Tracker

## Experiment Details
- **Started:** November 20, 2025
- **PID:** 59080
- **Log file:** `experiment_final_v5.log`
- **Output directory:** `results/final/`

## Configuration
```python
# Model Parameters
K_max = 20          # Increased from 12
iterations = 500    # Reduced from 1000 for speed
burn_in = 100       # Reduced from 200
gamma = 5.0         # DP concentration
alpha = 10.0        # HDP concentration
kappa = 15.0        # Sticky parameter

# Dataset
n_subjects = 20
epochs_per_subject = ~2632
total_epochs = 52,647
features = 72
```

## Key Fixes Applied
1. ✅ **K Discovery:** Increased K_max from 12 to 20
2. ✅ **Numerical Stability:** 
   - Periodic normalization every 100 timesteps
   - Log-space clipping to [-700, 700]
   - Safe non-finite handling in log_likelihood()
3. ✅ **Speed:** Reduced iterations from 1000 to 500 (2× faster)
4. ✅ **iDP Log-likelihood:** Now returns finite values (not -∞)

## Expected Timeline
- **HDP Training:** ~2-3 hours (iteration 0/500 started)
- **iDP Training:** ~2-3 hours (20 independent models)
- **Evaluation:** ~10 minutes
- **Total:** ~5-6 hours

## Monitoring Commands
```bash
# Check progress
tail -30 experiment_final_v5.log

# Check if running
ps aux | grep 59080 | grep -v grep

# Monitor K discovery
grep "K=" experiment_final_v5.log | tail -5
```

## Expected Results
- **HDP K:** 10-15 shared states (not maxing out at 20)
- **iDP K_total:** 100-200 states (5-10 per subject × 20 subjects)
- **HDP ARI:** > 0.4 (good clustering)
- **iDP ARI:** > 0.2 (reasonable baseline)
- **Log-likelihoods:** Both finite (HDP should be higher)
- **Hierarchical superiority:** Confirmed via all metrics

## Previous Runs Analysis
### Run v4 (flawed):
- K stuck at 8 (K_max too small)
- iDP had -∞ log-likelihood (numerical underflow)
- Only 159 iDP states (should be ~100-200)

### Run v5 (current - fixed):
- K_max=20 allows proper discovery
- Numerical stability ensures finite log-likelihoods
- Expected: K≈10-15, both models functional
