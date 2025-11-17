"""
Feature extraction for sleep staging.

Extracts spectral features from EEG/EOG/EMG signals.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import Dict, List, Optional, Tuple


# Frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 16.0),  # Sleep spindles
    'beta': (16.0, 30.0),
}


def compute_psd(
    epochs: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density for each epoch using Welch's method.
    
    Args:
        epochs: Array of shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for Welch's method
        noverlap: Number of overlapping samples (default: nperseg // 2)
    
    Returns:
        freqs: Frequency array
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    n_epochs, n_channels, n_samples = epochs.shape
    
    # Compute PSD for first epoch to get frequency array
    freqs, _ = signal.welch(
        epochs[0, 0],
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    n_freqs = len(freqs)
    psd = np.zeros((n_epochs, n_channels, n_freqs))
    
    for i in range(n_epochs):
        for j in range(n_channels):
            _, psd[i, j] = signal.welch(
                epochs[i, j],
                fs=sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
    
    return freqs, psd


def compute_bandpower(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float],
    log_transform: bool = True
) -> np.ndarray:
    """
    Compute total power in a frequency band.
    
    Args:
        freqs: Frequency array
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
        band: Frequency band as (low, high) in Hz
        log_transform: Apply log10 transform
    
    Returns:
        bandpower: Array of shape (n_epochs, n_channels)
    """
    # Find frequency indices for band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    # Integrate power in band (trapezoidal rule)
    freq_res = freqs[1] - freqs[0]
    bandpower = np.trapz(psd[:, :, idx_band], dx=freq_res, axis=-1)
    
    if log_transform:
        bandpower = np.log10(bandpower + 1e-10)  # Add epsilon to avoid log(0)
    
    return bandpower


def extract_spectral_features(
    epochs: np.ndarray,
    sampling_rate: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    log_transform: bool = True,
    nperseg: int = 256
) -> Dict[str, np.ndarray]:
    """
    Extract spectral features from epochs.
    
    Args:
        epochs: Array of shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of frequency bands (default: FREQ_BANDS)
        log_transform: Apply log10 transform to bandpowers
        nperseg: Segment length for Welch's method
    
    Returns:
        Dictionary of features:
            - 'psd': Full PSD (n_epochs, n_channels, n_freqs)
            - 'freqs': Frequency array
            - '<band>_power': Bandpower for each band (n_epochs, n_channels)
    """
    if bands is None:
        bands = FREQ_BANDS
    
    # Compute PSD
    freqs, psd = compute_psd(epochs, sampling_rate, nperseg=nperseg)
    
    features = {
        'psd': psd,
        'freqs': freqs,
    }
    
    # Compute bandpowers
    for band_name, band_range in bands.items():
        bandpower = compute_bandpower(freqs, psd, band_range, log_transform)
        features[f'{band_name}_power'] = bandpower
    
    return features


def extract_time_domain_features(
    epochs: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Extract simple time-domain features.
    
    Args:
        epochs: Array of shape (n_epochs, n_channels, n_samples)
    
    Returns:
        Dictionary of features (each of shape (n_epochs, n_channels)):
            - 'mean': Mean amplitude
            - 'std': Standard deviation
            - 'variance': Variance
    """
    return {
        'mean': np.mean(epochs, axis=-1),
        'std': np.std(epochs, axis=-1),
        'variance': np.var(epochs, axis=-1),
    }


def extract_features(
    epochs: np.ndarray,
    sampling_rate: float,
    include_spectral: bool = True,
    include_time_domain: bool = False,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    log_transform: bool = True
) -> np.ndarray:
    """
    Extract all features and concatenate into feature matrix.
    
    Args:
        epochs: Array of shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        include_spectral: Include spectral features (bandpowers)
        include_time_domain: Include time-domain features
        bands: Dictionary of frequency bands
        log_transform: Apply log10 transform to bandpowers
    
    Returns:
        Feature matrix of shape (n_epochs, n_features)
    """
    feature_list = []
    
    if include_spectral:
        spectral_features = extract_spectral_features(
            epochs, sampling_rate, bands, log_transform
        )
        
        # Concatenate bandpowers across channels
        for band_name in (bands or FREQ_BANDS).keys():
            bandpower = spectral_features[f'{band_name}_power']  # (n_epochs, n_channels)
            # Flatten channels into features
            feature_list.append(bandpower.reshape(len(bandpower), -1))
    
    if include_time_domain:
        time_features = extract_time_domain_features(epochs)
        for feat_name, feat_values in time_features.items():
            feature_list.append(feat_values.reshape(len(feat_values), -1))
    
    # Concatenate all features
    X = np.concatenate(feature_list, axis=1)
    
    return X


def preprocess_subject(
    epochs: np.ndarray,
    sampling_rate: float,
    standardize: bool = True,
    **feature_kwargs
) -> np.ndarray:
    """
    Complete preprocessing pipeline for one subject.
    
    Args:
        epochs: Raw epochs (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        standardize: Apply z-score standardization
        **feature_kwargs: Additional arguments for extract_features
    
    Returns:
        Feature matrix (n_epochs, n_features)
    """
    # Extract features
    X = extract_features(epochs, sampling_rate, **feature_kwargs)
    
    # Standardize
    if standardize:
        X = zscore(X, axis=0, nan_policy='omit')
        # Replace NaNs with 0 (can happen if feature has zero variance)
        X = np.nan_to_num(X, nan=0.0)
    
    return X


def main():
    """Test feature extraction."""
    # Generate synthetic data
    n_epochs = 100
    n_channels = 2
    n_samples = 3000  # 30 seconds at 100 Hz
    sampling_rate = 100.0
    
    np.random.seed(42)
    epochs = np.random.randn(n_epochs, n_channels, n_samples)
    
    # Extract features
    X = preprocess_subject(epochs, sampling_rate)
    
    print(f"Input shape: {epochs.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features per channel: {X.shape[1] // n_channels}")
    print(f"Feature statistics:")
    print(f"  Mean: {X.mean():.3f}")
    print(f"  Std: {X.std():.3f}")
    print(f"  Min: {X.min():.3f}")
    print(f"  Max: {X.max():.3f}")


if __name__ == "__main__":
    main()
