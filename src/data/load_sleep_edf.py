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
    
    raw.pick_channels(available_channels)
    
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
    
    # Extract 30-second epochs
    epoch_duration = 30.0  # seconds
    n_epochs = int(raw.times[-1] / epoch_duration)
    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_duration * sfreq)
    
    # Extract spectral features for each epoch
    features_list = []
    labels_list = []
    
    for i in range(n_epochs):
        start_sample = i * samples_per_epoch
        end_sample = start_sample + samples_per_epoch
        
        if end_sample > len(raw.times):
            break
        
        # Get data for this epoch
        epoch_data = raw.get_data(start=start_sample, stop=end_sample)
        
        # Compute spectral power features (simple frequency bands)
        features = compute_spectral_features(epoch_data, sfreq)
        
        # Get corresponding sleep stage
        epoch_time = i * epoch_duration
        
        # Find annotation at this time
        stage_label = -1
        for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
            if onset <= epoch_time < onset + duration:
                if desc in stage_mapping:
                    stage_label = stage_mapping[desc]
                break
        
        if stage_label >= 0:  # Valid sleep stage
            features_list.append(features)
            labels_list.append(stage_label)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
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


def load_sleep_edf_dataset(
    data_dir: Path,
    n_subjects: int = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multiple subjects from Sleep-EDF dataset.
    
    Args:
        data_dir: Path to data/raw/sleep-cassette directory
        n_subjects: Number of subjects to load (None = all available)
        verbose: Print progress
    
    Returns:
        X_list: List of feature arrays, one per subject
        y_list: List of label arrays, one per subject
    """
    data_dir = Path(data_dir)
    
    # Find all PSG files
    psg_files = sorted(data_dir.glob('*-PSG.edf'))
    
    if n_subjects is not None:
        psg_files = psg_files[:n_subjects]
    
    X_list = []
    y_list = []
    
    for psg_file in psg_files:
        # Find corresponding hypnogram (PSG: SC4001E0-PSG.edf -> Hypno: SC4001EC-Hypnogram.edf)
        # Hypnogram suffixes vary: EC, EH, EJ, EP, EU, EV, etc.
        subject_base = psg_file.stem.replace('-PSG', '').replace('E0', '')  # e.g., SC4001
        
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
        
        if verbose:
            print(f"  Loading {psg_file.stem.replace('-PSG', '')}...")
        
        try:
            X, y = load_sleep_edf_subject(psg_file, hypno_file)
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            if verbose:
                print(f"  Error loading {subject_id}: {e}")
            continue
    
    if verbose:
        print(f"  Loaded {len(X_list)} subjects")
        print(f"  Avg epochs per subject: {np.mean([len(X) for X in X_list]):.0f}")
        print(f"  Features per epoch: {X_list[0].shape[1]}")
    
    return X_list, y_list


if __name__ == '__main__':
    # Test loading
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sleep-cassette'
    X_list, y_list = load_sleep_edf_dataset(data_dir, n_subjects=2, verbose=True)
    
    print("\nFirst subject:")
    print(f"  Shape: {X_list[0].shape}")
    print(f"  Sleep stages: {np.unique(y_list[0])}")
    print(f"  Stage distribution: {np.bincount(y_list[0])}")
