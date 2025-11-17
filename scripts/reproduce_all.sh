#!/bin/bash
# Complete reproduction script for all experiments

set -e  # Exit on error

echo "========================================"
echo "Sleep-EDF HDP-HMM Reproduction Script"
echo "========================================"

# Configuration
N_SUBJECTS=30
COHORT="cassette"
OUTPUT_DIR="results/paper"
RANDOM_SEED=42

echo ""
echo "Step 1: Download data"
python3 scripts/download_data.py \
    --output data/raw \
    --n-subjects $N_SUBJECTS \
    --cohort $COHORT \
    --nights 1

echo ""
echo "Step 2: Preprocess features"
python3 scripts/preprocess_all.py \
    --input data/raw \
    --output data/processed \
    --features psd bandpower

echo ""
echo "Step 3: Run independent DP-HMM baseline"
python3 scripts/run_idp_hmm.py \
    --data data/processed \
    --config configs/default_config.yaml \
    --output results/idp_hmm \
    --seed $RANDOM_SEED

echo ""
echo "Step 4: Run sticky HDP-HMM"
python3 scripts/run_hdp_hmm_sticky.py \
    --data data/processed \
    --config configs/default_config.yaml \
    --output results/hdp_hmm_sticky \
    --seed $RANDOM_SEED

echo ""
echo "Step 5: Run LOSO cross-validation"
python3 scripts/run_loso_cv.py \
    --data data/processed \
    --models idp_hmm hdp_hmm_sticky \
    --n-folds $N_SUBJECTS \
    --output results/loso_cv \
    --seed $RANDOM_SEED

echo ""
echo "Step 6: Generate figures and tables"
python3 scripts/generate_figures.py \
    --input results/loso_cv \
    --output $OUTPUT_DIR/figures

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
