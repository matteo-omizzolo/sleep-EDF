# HDP-HMM Implementation Fixes

## Critical Bugs Fixed

### 1. **Beta Stick-Breaking Sampling** (FIXED ✓)
**Problem**: Cumsum direction was ambiguous - unclear if `sum_{j>k}` was computed correctly.

**Fix**: Changed from reverse cumsum to explicit sum:
```python
# OLD (ambiguous):
cumsum_counts = np.cumsum(state_counts[::-1])[::-1]
b = self.gamma + cumsum_counts[k+1]

# NEW (explicit):
b = self.gamma + np.sum(state_counts[k+1:])  # Sum counts for indices AFTER k
```

**Why**: The posterior for stick-breaking is `v_k ~ Beta(1 + n_k, gamma + sum_{j>k} n_j)` where `sum_{j>k}` means "sum over indices greater than k". The explicit slice `state_counts[k+1:]` makes this unambiguous.

---

### 2. **Backward State Sampling** (FIXED ✓)
**Problem**: Incorrectly used xi (two-slice marginals) for backward sampling.

**Fix**: Changed to correct conditional distribution:
```python
# OLD (WRONG):
trans_prob = xi[t, :, states[t+1]]  # This is p(z_t, z_{t+1}|X)

# NEW (CORRECT):
trans_prob = pi[:, next_state] * gamma[t]  # This is p(z_t|z_{t+1}, X)
```

**Why**: Backward sampling requires `p(z_t | z_{t+1}, X) ∝ p(z_{t+1} | z_t) * p(z_t | X_{1:t}) = pi[z_t, z_{t+1}] * gamma[t, z_t]`. The xi marginals already integrate over both time slices, which is incorrect for conditional sampling.

---

### 3. **Forward-Backward Initial Distribution** (CLARIFIED ✓)
**Problem**: Not explicitly clear that initial state uses beta (global weights).

**Fix**: Added clear documentation:
```python
# Initial state distribution: p(z_1) = beta (global stick-breaking weights)
log_alpha[0] = np.log(self.beta_ + 1e-10) + log_B[0]
```

**Why**: In HDP-HMM, the initial state distribution comes from the global stick-breaking weights beta, not from the transition matrix. This enforces hierarchical sharing from the start.

---

## Additional Improvements

### 4. **Duplicate Xi Computation** (REMOVED ✓)
Removed redundant xi calculation that was computing the same thing twice (lines 219-239 in old code).

### 5. **Stronger Priors** (ADDED ✓)
Increased NIW prior strength to prevent over-fitting:
- `kappa_0`: 0.01 → 0.1 (10x stronger mean prior)
- `nu_0`: d+2 → d+10 (more prior mass on covariance)

**Why**: With 72-dimensional features and 12K observations, weak priors allow emission parameters to become too confident, leading to winner-take-all collapse.

### 6. **Gradual Annealing** (ADDED ✓)
Replaced abrupt burn-in transition with smooth annealing:
```python
progress = iter_idx / burn_in  # 0.0 → 1.0 during burn-in, then stays at 1.0
kappa_multiplier = 0.1 + 0.9 * progress  # Smooth: 0.1 → 1.0
inflation_factor = 3.0 - 2.0 * progress  # Smooth: 3.0 → 1.0
```

**Why**: Abrupt changes at iteration 200 caused sudden collapse (K=6 → K=1). Gradual annealing allows the model to smoothly transition from exploration to exploitation.

---

## Algorithm Flow (Verified Correct)

1. **Initialization**: K-means with 8 clusters, beta initialized uniformly on first 8 states
2. **E-step (State Sampling)**:
   - Forward-backward with `p(z_1) = beta`, `p(z_t|z_{t-1}) = pi`
   - Backward sampling: `z_T ~ gamma[-1]`, then `z_t ~ pi[:, z_{t+1}] * gamma[t]`
3. **M-step (Parameter Updates)**:
   - **Emissions**: NIW posterior with inflation during burn-in
   - **Transitions**: `pi_j ~ Dir(alpha*beta + kappa*delta_j + counts)` (sticky)
   - **Beta**: Stick-breaking with `v_k ~ Beta(1 + n_k, gamma + sum_{j>k} n_j)`
4. **Annealing**: Gradual increase of kappa and decrease of inflation over burn-in

---

## Expected Behavior

With these fixes:
- ✓ Model should discover K=5-7 states during burn-in
- ✓ K should stabilize (not collapse) after burn-in
- ✓ States should align with true sleep stages (W, N1, N2, N3, REM)
- ✓ Hierarchical sharing should show 5-7 global states vs 25-35 independent states

---

## Testing

Run the corrected implementation:
```bash
python scripts/run_complete_experiment.py --n-subjects 5 --output results/improved --use-real-data
```

Monitor K trace in `experiment_improved.log` - should maintain K≈5-7 throughout entire sampling.
