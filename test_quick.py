#!/usr/bin/env python3
"""Quick test to verify installation."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing imports...")
import numpy as np
print("✓ numpy")

from models.simple_hdp_hmm import SimpleStickyHDPHMM
print("✓ SimpleStickyHDPHMM")

from eval.metrics import compute_ari
print("✓ metrics")

print("\nGenerating tiny dataset...")
X1 = np.random.randn(50, 5)
X2 = np.random.randn(50, 5)

print("Fitting model...")
model = SimpleStickyHDPHMM(
    K_max=5,
    n_iter=10,
    burn_in=5,
    verbose=1
)
model.fit([X1, X2])

print("\n✓ SUCCESS! Everything is working.")
print(f"Model found K={model.get_posterior_mean_K():.1f} states")
