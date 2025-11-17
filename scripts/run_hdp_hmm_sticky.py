#!/usr/bin/env python3
"""
Run sticky HDP-HMM on Sleep-EDF data.

Usage:
    python scripts/run_hdp_hmm_sticky.py \
        --data data/processed \
        --config configs/default_config.yaml \
        --output results/hdp_hmm
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from models.hdp_hmm_sticky import StickyHDPHMM


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Sticky HDP-HMM on Sleep-EDF data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hdp_hmm",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 60)
    print("Sticky HDP-HMM for Sleep Staging")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (placeholder - you would implement data loading here)
    print("\n[1/4] Loading data...")
    # X_train, y_train = load_data(args.data, 'train')
    # X_test, y_test = load_data(args.data, 'test')
    
    # For demonstration, create dummy data
    n_subjects = 5
    n_epochs_per_subject = 800
    n_features = 10
    
    X_train = [np.random.randn(n_epochs_per_subject, n_features) for _ in range(n_subjects)]
    
    print(f"  Loaded {len(X_train)} subjects")
    print(f"  Features: {n_features}")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model_config = config['model']
    
    model = StickyHDPHMM(
        gamma=model_config['concentration']['gamma']['shape'],
        alpha=model_config['concentration']['alpha']['shape'],
        kappa=model_config['concentration']['kappa']['shape'],
        K_max=config['inference']['K_max'],
        n_iter=config['inference']['n_iter'],
        burn_in=config['inference']['burn_in'],
        thin=config['inference']['thin'],
        random_state=args.seed,
        verbose=1,
    )
    
    # Fit model
    print("\n[3/4] Fitting model...")
    model.fit(X_train)
    
    # Save model
    print("\n[4/4] Saving results...")
    model.save(output_dir / "model.pkl")
    print(f"  Model saved to {output_dir / 'model.pkl'}")
    
    # Save summary statistics
    summary = {
        'K_mean': model.K_,
        'n_subjects': len(X_train),
        'n_features': n_features,
        'config': config,
    }
    
    with open(output_dir / "summary.yaml", 'w') as f:
        yaml.dump(summary, f)
    
    print(f"  Summary saved to {output_dir / 'summary.yaml'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
