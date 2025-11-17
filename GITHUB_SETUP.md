# GitHub Repository Setup Instructions

Your local git repository has been initialized! Here's how to push it to GitHub:

## Option 1: Using GitHub CLI (Recommended)

```bash
# Navigate to project directory
cd "/Users/matteoomizzolo/Desktop/Github Projects/BayesianNonParametrics/sleep-EDF"

# Create GitHub repository and push
gh repo create sleep-EDF --public --source=. --remote=origin --push
```

## Option 2: Using GitHub Web Interface

### 1. Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `sleep-EDF`
3. Description: "Hierarchical Dirichlet Process HMM for unsupervised sleep staging on PhysioNet Sleep-EDF data"
4. Choose: **Public** (or Private if preferred)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 2. Push Local Repository

```bash
# Navigate to project directory
cd "/Users/matteoomizzolo/Desktop/Github Projects/BayesianNonParametrics/sleep-EDF"

# Add GitHub remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/sleep-EDF.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Option 3: Using SSH (if you have SSH keys configured)

```bash
cd "/Users/matteoomizzolo/Desktop/Github Projects/BayesianNonParametrics/sleep-EDF"

# Add remote with SSH
git remote add origin git@github.com:YOUR-USERNAME/sleep-EDF.git

# Push
git branch -M main
git push -u origin main
```

## Verify Repository Status

Check what's been committed:

```bash
git log --oneline
git status
```

## Next Steps After Pushing

1. **Add repository topics** on GitHub:
   - bayesian-nonparametrics
   - hdp-hmm
   - sleep-staging
   - polysomnography
   - machine-learning
   - python

2. **Enable GitHub Pages** (optional) for documentation:
   - Settings → Pages → Source: Deploy from branch → main → /docs

3. **Add repository description**:
   "Sticky HDP-HMM for unsupervised sleep stage segmentation using PhysioNet Sleep-EDF data"

4. **Create a GitHub issue** for TODO items:
   - Implement full MCMC parameter updates
   - Add Viterbi decoding
   - Create visualization notebooks
   - Add unit tests

## Project Structure Created

```
sleep-EDF/
├── README.md                    ✓ Comprehensive documentation
├── QUICKSTART.md               ✓ Quick start guide
├── LICENSE                      ✓ MIT License
├── requirements.txt            ✓ Python dependencies
├── setup.py                     ✓ Package installer
├── .gitignore                   ✓ Git ignore rules
├── configs/
│   └── default_config.yaml     ✓ Default hyperparameters
├── src/
│   ├── data/                    ✓ Data loading & preprocessing
│   ├── models/                  ✓ HDP-HMM implementation
│   └── eval/                    ✓ Evaluation metrics
└── scripts/
    ├── download_data.py        ✓ PhysioNet downloader
    ├── run_hdp_hmm_sticky.py   ✓ Main experiment script
    └── reproduce_all.sh        ✓ Full reproduction script
```

## Current Commit

```
commit 06741c3
Initial commit: Sticky HDP-HMM for sleep staging

- Implemented sticky HDP-HMM model with hierarchical state sharing
- Data loading and preprocessing for Sleep-EDF dataset
- Evaluation metrics (ARI, NMI, F1, Hungarian alignment)
- Configuration system with YAML
- Scripts for downloading data and running experiments
- Comprehensive README with project documentation
- MIT License
```

Your repository is ready to push to GitHub! 🚀
