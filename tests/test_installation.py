"""
Test basic functionality to verify installation.

Run with: python -m pytest tests/ -v
Or just: python tests/test_installation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        from models.base import BaseHMM
        print("  ✓ models.base")
    except ImportError as e:
        print(f"  ✗ models.base: {e}")
        return False
    
    try:
        from models.hdp_hmm_sticky import StickyHDPHMM
        print("  ✓ models.hdp_hmm_sticky")
    except ImportError as e:
        print(f"  ✗ models.hdp_hmm_sticky: {e}")
        return False
    
    try:
        from eval.metrics import compute_ari
        print("  ✓ eval.metrics")
    except ImportError as e:
        print(f"  ✗ eval.metrics: {e}")
        return False
    
    try:
        from eval.hungarian import hungarian_alignment
        print("  ✓ eval.hungarian")
    except ImportError as e:
        print(f"  ✗ eval.hungarian: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that we can create a model instance."""
    print("\nTesting model creation...")
    
    try:
        from models.hdp_hmm_sticky import StickyHDPHMM
        
        model = StickyHDPHMM(
            gamma=1.0,
            alpha=1.0,
            kappa=5.0,
            K_max=20,
            random_state=42
        )
        
        print(f"  ✓ Created StickyHDPHMM with K_max={model.K_max}")
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_data_processing():
    """Test basic data processing functions."""
    print("\nTesting data processing...")
    
    try:
        import numpy as np
        from data.preprocessing import extract_features
        
        # Create dummy epoch data
        n_epochs = 10
        n_channels = 2
        n_samples = 3000  # 30 seconds at 100 Hz
        sampling_rate = 100.0
        
        epochs = np.random.randn(n_epochs, n_channels, n_samples)
        
        # Extract features
        X = extract_features(epochs, sampling_rate)
        
        print(f"  ✓ Extracted features: {X.shape}")
        print(f"    Input: {epochs.shape} → Output: {X.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data processing failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    try:
        import numpy as np
        from eval.metrics import compute_ari, compute_nmi
        from eval.hungarian import hungarian_alignment
        
        # Create dummy labels
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 10)
        y_pred = np.array([1, 1, 2, 2, 3, 3, 4, 4, 0, 0] * 10)
        
        # Compute metrics
        ari = compute_ari(y_true, y_pred)
        nmi = compute_nmi(y_true, y_pred)
        
        print(f"  ✓ ARI: {ari:.3f}")
        print(f"  ✓ NMI: {nmi:.3f}")
        
        # Test Hungarian alignment
        y_aligned, mapping = hungarian_alignment(y_true, y_pred)
        print(f"  ✓ Hungarian alignment: {mapping}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Metrics failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Sleep-EDF HDP-HMM Installation Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Data Processing", test_data_processing()))
    results.append(("Metrics", test_metrics()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! Installation is working correctly. ✓")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
