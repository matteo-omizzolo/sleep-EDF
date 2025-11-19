#!/usr/bin/env python3
"""
Load and preprocess Sleep-EDF data from PhysioNet.

The Sleep-EDF database contains whole-night polysomnographic sleep recordings,
containing EEG, EOG, chin EMG, and event markers. Sleep stages are annotated
as: W (wake), N1, N2, N3, REM.
"""

import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_sleep_edf_subject(
    psg_file: Path,
    hypnogram_file: Path,
    target_channels: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one subject's PSG data and hypnogram annotations.
    
    Args:
        psg_file: Path to PSG .edf file
        hypnogram_file: Path to Hypnogram .edf file
        target_channels: List of channel names to extract (default: Fpz-Cz, Pz-Oz)
    
    Returns:
        X: Features (n_epochs, n_features) - spectral power features per 30-sec epoch
        y: True sleep stages (n_epochs,) - integers 0-4 for W/N1/N2/N3/REM
    """
    if target_channels is None:
        target_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    
    # Load PSG signals
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    
    # Select EEG channels
    available_channels = [ch for ch in target_channels if ch in raw.ch_names]
    if not available_channels:
        # Fallback to any EEG channels
        available_channels = [ch for ch in raw.ch_names if 'EEG' in ch][:2]
    
    # Use new API to avoid legacy warning
    raw.pick(available_channels)
    
    # Load hypnogram (sleep stage annotations)
    annot = mne.read_annotations(hypnogram_file)
    
    # Map sleep stages to integers
    stage_mapping = {
        'Sleep stage W': 0,      # Wake
        'Sleep stage 1': 1,      # N1
        'Sleep stage 2': 2,      # N2
        'Sleep stage 3': 3,      # N3
        'Sleep stage 4': 3,      # N3 (combine S3 and S4)
        'Sleep stage R': 4,      # REM
        'Sleep stage ?': -1,     # Unknown (will filter out)
        'Movement time': -1,     # Movement (will filter out)
    }
    
    # Extract 30-second epochs (vectorized)
    epoch_duration = 30.0  # seconds
    sfreq = float(raw.info['sfreq'])
    samples_per_epoch = int(epoch_duration * sfreq)

    full_data = raw.get_data()  # shape (n_channels, n_samples)
    n_samples = full_data.shape[1]
    n_epochs = n_samples // samples_per_epoch
    if n_epochs == 0:
        return np.empty((0, len(available_channels) * 5)), np.empty((0,), dtype=int)

    trimmed = full_data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(trimmed.shape[0], n_epochs, samples_per_epoch)

    X_all = compute_spectral_features_batch(epochs, sfreq, add_temporal=True)

    # Vectorized label assignment per epoch from annotations
    y = np.full(n_epochs, -1, dtype=int)
    if len(annot.onset) > 0:
        for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
            code = stage_mapping.get(desc, -1)
            if code < 0:
                continue
            start_idx = int(max(0, np.floor(onset / epoch_duration)))
            end_idx = int(np.ceil((onset + duration) / epoch_duration))
            if start_idx < n_epochs:
                y[start_idx:min(end_idx, n_epochs)] = code

    valid = y >= 0
    X = X_all[valid]
    y = y[valid]

    return X, y


def compute_spectral_features(
    data: np.ndarray,
    sfreq: float
) -> np.ndarray:
    """
    Compute spectral power features from raw EEG.
    
    Computes power in standard frequency bands:
    - Delta (0.5-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma (30-50 Hz)
    
    Args:
        data: Raw EEG (n_channels, n_samples)
        sfreq: Sampling frequency
    
    Returns:
        features: Array of shape (n_channels * 5,) containing band powers
    """
    from scipy import signal
    
    n_channels = data.shape[0]
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    features = []
    
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx]
        
        # Compute power spectral density
        freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(256, len(ch_data)))
        
        # Extract power in each band
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.log(np.mean(psd[band_mask]) + 1e-10)
            features.append(band_power)
    
    return np.array(features)


def add_temporal_context_features(X: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Add temporal context features (derivatives and rolling statistics).
    
    Critical for sleep staging as sleep stages evolve over time!
    
    Args:
        X: Feature array (n_epochs, n_features)
        window: Rolling window size (default 3 = 90 seconds)
    
    Returns:
        X_enhanced: Array with added temporal features
    """
    n_epochs, n_features = X.shape
    
    # First-order derivatives (rate of change)
    X_diff1 = np.zeros_like(X)
    X_diff1[1:] = np.diff(X, axis=0)
    X_diff1[0] = X_diff1[1]  # Copy first value
    
    # Rolling mean (smoothed features)
    from scipy.ndimage import uniform_filter1d
    X_rolling_mean = uniform_filter1d(X, size=window, axis=0, mode='nearest')
    
    # Rolling std (variability over time)
    X_rolling_std = np.zeros_like(X)
    for i in range(n_epochs):
        start = max(0, i - window // 2)
        end = min(n_epochs, i + window // 2 + 1)
        X_rolling_std[i] = np.std(X[start:end], axis=0)
    
    # Concatenate all features
    X_enhanced = np.concatenate([
        X,                  # Original features
        X_diff1,            # Temporal derivatives
        X_rolling_mean,     # Smoothed context
        X_rolling_std       # Local variability
    ], axis=1)
    
    return X_enhanced


def compute_spectral_features_batch(
    epochs: np.ndarray,
    sfreq: float,
    add_temporal: bool = True
) -> np.ndarray:
    """
    Compute comprehensive spectral and temporal features for all epochs at once.
    
    Features include:
    - Band powers (delta, theta, alpha, beta, gamma) - 5 per channel
    - Spectral ratios (theta/alpha, alpha/delta) - 2 per channel  
    - Temporal features (variance, mobility) - 2 per channel
    Base: 9 features per channel
    
    If add_temporal=True (default), also adds:
    - First derivatives (9 per channel)
    - Rolling mean (9 per channel)
    - Rolling std (9 per channel)
    Total with temporal: 36 features per channel

    Args:
        epochs: Array of shape (n_channels, n_epochs, n_samples)
        sfreq: Sampling frequency
        add_temporal: Whether to add temporal context features

    Returns:
        X: Array of shape (n_epochs, n_channels * 9 * 4) if add_temporal
           or (n_epochs, n_channels * 9) otherwise
    """
    from scipy import signal

    n_channels, n_epochs, n_samples = epochs.shape

    bands = [
        ('delta', 0.5, 4),
        ('theta', 4, 8),
        ('alpha', 8, 13),
        ('beta', 13, 30),
        ('gamma', 30, 50),
    ]

    # Welch PSD across the last axis (samples) for all channels and epochs
    freqs, psd = signal.welch(epochs, sfreq, nperseg=min(256, n_samples), axis=-1)
    # psd shape: (n_channels, n_epochs, n_freqs)

    # Precompute band masks
    masks = [(freqs >= low) & (freqs <= high) for _, low, high in bands]

    # Compute band powers per (channel, epoch)
    band_powers = []  # list of arrays shape (n_channels, n_epochs)
    for mask in masks:
        power = np.log(psd[..., mask].mean(axis=-1) + 1e-10)
        band_powers.append(power)
    
    # Compute spectral ratios (important for sleep staging)
    theta_power = band_powers[1]  # theta is index 1
    alpha_power = band_powers[2]  # alpha is index 2
    delta_power = band_powers[0]  # delta is index 0
    
    theta_alpha_ratio = theta_power / (alpha_power + 1e-10)
    alpha_delta_ratio = alpha_power / (delta_power + 1e-10)
    
    # Compute temporal features (Hjorth parameters)
    # Variance (activity)
    variance = np.log(np.var(epochs, axis=-1) + 1e-10)  # (n_channels, n_epochs)
    
    # Mobility (measure of signal's mean frequency)
    diff_data = np.diff(epochs, axis=-1)
    mobility = np.sqrt(np.var(diff_data, axis=-1) / (np.var(epochs, axis=-1) + 1e-10))

    # Assemble features with channel-major ordering:
    # [ch0_features, ch1_features, ...]
    features_per_epoch = []
    for ch in range(n_channels):
        ch_feats = np.stack([
            band_powers[0][ch, :],  # delta
            band_powers[1][ch, :],  # theta
            band_powers[2][ch, :],  # alpha
            band_powers[3][ch, :],  # beta
            band_powers[4][ch, :],  # gamma
            theta_alpha_ratio[ch, :],
            alpha_delta_ratio[ch, :],
            variance[ch, :],
            mobility[ch, :],
        ], axis=1)  # (n_epochs, 9)
        features_per_epoch.append(ch_feats)

    X = np.concatenate(features_per_epoch, axis=1)  # (n_epochs, n_channels*9)
    
    # Add temporal context features if requested
    if add_temporal:
        X = add_temporal_context_features(X, window=3)
    
    return X


def load_sleep_edf_dataset(
    data_dir: Path,
    n_subjects: int = None,
    verbose: bool = True,
    use_cache: bool = True,
    incremental_cache: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multiple subjects from Sleep-EDF dataset.
    
    Args:
        data_dir: Path to data/raw/sleep-cassette directory
        n_subjects: Number of subjects to load (None = all available)
        verbose: Print progress
        use_cache: Use cached preprocessed features if available
    
    Returns:
        X_list: List of feature arrays, one per subject
        y_list: List of label arrays, one per subject
    """
    data_dir = Path(data_dir)
    
    # Check for cached data
    cache_dir = data_dir.parent.parent / 'processed'
    cache_file = cache_dir / f'sleep_edf_cache_{n_subjects}subj.npz'
    subj_cache_dir = cache_dir / 'sleep_edf_subjects'
    
    if use_cache and not incremental_cache and cache_file.exists():
        if verbose:
            print(f"  Loading cached data from {cache_file.name}...")
        data = np.load(cache_file, allow_pickle=True)
        X_list = list(data['X_list'])
        y_list = list(data['y_list'])
        if verbose:
            print(f"  Loaded {len(X_list)} subjects from cache")
        return X_list, y_list
    
    # Find all PSG files
    psg_files = sorted(data_dir.glob('*-PSG.edf'))
    
    if n_subjects is not None:
        psg_files = psg_files[:n_subjects]
    
    X_list = []
    y_list = []
    
    for psg_file in psg_files:
        # Find corresponding hypnogram (PSG: SC4001E0-PSG.edf -> Hypno: SC4001EC-Hypnogram.edf)
        # Hypnogram suffixes vary: EC, EH, EJ, EP, EU, EV, etc.
        subject_id = psg_file.stem.replace('-PSG', '')        # e.g., SC4001E0
        subject_base = subject_id.replace('E0', '')           # e.g., SC4001
        
        # Try common hypnogram suffixes
        possible_suffixes = ['EC', 'EH', 'EJ', 'EP', 'EU', 'EV', 'EA', 'EM', 'EW', 'EG']
        hypno_file = None
        
        for suffix in possible_suffixes:
            candidate = psg_file.parent / f"{subject_base}{suffix}-Hypnogram.edf"
            if candidate.exists():
                hypno_file = candidate
                break
        
        if not hypno_file or not hypno_file.exists():
            if verbose:
                print(f"  Warning: No hypnogram for {psg_file.stem}, skipping")
            continue
        
        # Per-subject caching path
        subj_cache_path = subj_cache_dir / f"{subject_id}.npz"

        if use_cache and incremental_cache and subj_cache_path.exists():
            if verbose:
                print(f"  Loading {subject_id} from cache...")
            try:
                data = np.load(subj_cache_path, allow_pickle=True)
                X = data['X']
                y = data['y']
                X_list.append(X)
                y_list.append(y)
                continue
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to read cache for {subject_id}: {e}. Recomputing...")
                # fall through to recompute

        if verbose:
            print(f"  Loading {subject_id}...")

        try:
            X, y = load_sleep_edf_subject(psg_file, hypno_file)
            X_list.append(X)
            y_list.append(y)

            # Save per-subject cache immediately
            if use_cache and incremental_cache:
                subj_cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(subj_cache_path, X=X, y=y)
        except Exception as e:
            if verbose:
                print(f"  Error loading {subject_id}: {e}")
            continue
    
    if verbose:
        print(f"  Loaded {len(X_list)} subjects")
        print(f"  Avg epochs per subject: {np.mean([len(X) for X in X_list]):.0f}")
        print(f"  Features per epoch: {X_list[0].shape[1]}")
    
    # Save combined cache snapshot
    if use_cache and (not incremental_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Saving to cache: {cache_file.name}...")
        np.savez_compressed(cache_file, X_list=X_list, y_list=y_list)
    
    return X_list, y_list


if __name__ == '__main__':
    # Test loading
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sleep-cassette'
    X_list, y_list = load_sleep_edf_dataset(data_dir, n_subjects=2, verbose=True)
    
    print("\nFirst subject:")
    print(f"  Shape: {X_list[0].shape}")
    print(f"  Sleep stages: {np.unique(y_list[0])}")
    print(f"  Stage distribution: {np.bincount(y_list[0])}")
