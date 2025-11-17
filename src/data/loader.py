"""
EDF file loader for Sleep-EDF dataset.

Loads polysomnography signals and hypnogram labels.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mne


# Sleep stage mapping
STAGE_MAPPING = {
    'Sleep stage W': 0,  # Wake
    'Sleep stage 1': 1,  # N1
    'Sleep stage 2': 2,  # N2
    'Sleep stage 3': 3,  # N3 (deep sleep)
    'Sleep stage 4': 3,  # N3 (some datasets still use stage 4)
    'Sleep stage R': 4,  # REM
    'Sleep stage ?': -1,  # Unknown/artifact
    'Movement time': -1,  # Movement/artifact
}

STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']


def load_sleep_edf(
    psg_file: Path,
    channels: Optional[List[str]] = None,
    epoch_length: float = 30.0
) -> Tuple[np.ndarray, List[str], float]:
    """
    Load polysomnography signals from EDF file.
    
    Args:
        psg_file: Path to PSG EDF file
        channels: List of channel names to load (None = all)
        epoch_length: Epoch length in seconds
    
    Returns:
        signals: Array of shape (n_channels, n_samples)
        channel_names: List of channel names
        sampling_rate: Sampling rate in Hz
    """
    # Load EDF file with MNE
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    
    # Get sampling rate
    sampling_rate = raw.info['sfreq']
    
    # Select channels
    if channels is not None:
        # Try to find matching channels (case-insensitive)
        available_channels = raw.ch_names
        selected = []
        for ch in channels:
            matches = [c for c in available_channels if ch.lower() in c.lower()]
            if matches:
                selected.extend(matches)
        
        if not selected:
            raise ValueError(f"No matching channels found for {channels}")
        
        raw.pick_channels(selected)
    
    # Get data
    signals = raw.get_data()  # Shape: (n_channels, n_samples)
    channel_names = raw.ch_names
    
    return signals, channel_names, sampling_rate


def load_hypnogram(
    hypnogram_file: Path,
    epoch_length: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hypnogram (sleep stage annotations) from EDF file.
    
    Args:
        hypnogram_file: Path to Hypnogram EDF file
        epoch_length: Expected epoch length in seconds
    
    Returns:
        stages: Array of integer sleep stages (n_epochs,)
        times: Array of epoch start times in seconds (n_epochs,)
    """
    # Load annotations from hypnogram EDF
    annotations = mne.read_annotations(hypnogram_file)
    
    # Extract stage labels and times
    stages = []
    times = []
    
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        # Map stage description to integer
        stage = STAGE_MAPPING.get(description, -1)
        
        # Skip unknown stages
        if stage == -1:
            continue
        
        stages.append(stage)
        times.append(onset)
    
    return np.array(stages), np.array(times)


def segment_into_epochs(
    signals: np.ndarray,
    sampling_rate: float,
    epoch_length: float = 30.0
) -> np.ndarray:
    """
    Segment continuous signals into fixed-length epochs.
    
    Args:
        signals: Array of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        epoch_length: Epoch length in seconds
    
    Returns:
        epochs: Array of shape (n_epochs, n_channels, epoch_samples)
    """
    n_channels, n_samples = signals.shape
    epoch_samples = int(epoch_length * sampling_rate)
    n_epochs = n_samples // epoch_samples
    
    # Trim to complete epochs
    signals_trimmed = signals[:, :n_epochs * epoch_samples]
    
    # Reshape into epochs
    epochs = signals_trimmed.reshape(n_channels, n_epochs, epoch_samples)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, epoch_samples)
    
    return epochs


def load_subject_data(
    psg_file: Path,
    hypnogram_file: Path,
    channels: Optional[List[str]] = None,
    epoch_length: float = 30.0
) -> Dict:
    """
    Load complete subject data (signals + labels).
    
    Args:
        psg_file: Path to PSG EDF file
        hypnogram_file: Path to Hypnogram EDF file
        channels: List of channel names to load
        epoch_length: Epoch length in seconds
    
    Returns:
        Dictionary containing:
            - epochs: (n_epochs, n_channels, epoch_samples)
            - stages: (n_epochs,) integer labels
            - channel_names: list of channel names
            - sampling_rate: float
    """
    # Load signals
    signals, channel_names, sampling_rate = load_sleep_edf(
        psg_file, channels, epoch_length
    )
    
    # Load hypnogram
    stages, stage_times = load_hypnogram(hypnogram_file, epoch_length)
    
    # Segment signals into epochs
    epochs = segment_into_epochs(signals, sampling_rate, epoch_length)
    
    # Align epochs and stages (they should match, but check)
    n_epochs = min(len(epochs), len(stages))
    
    return {
        'epochs': epochs[:n_epochs],
        'stages': stages[:n_epochs],
        'channel_names': channel_names,
        'sampling_rate': sampling_rate,
        'epoch_length': epoch_length,
    }


def main():
    """Test loading."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python loader.py <psg_file> <hypnogram_file>")
        sys.exit(1)
    
    psg_file = Path(sys.argv[1])
    hypnogram_file = Path(sys.argv[2])
    
    data = load_subject_data(psg_file, hypnogram_file)
    
    print(f"Loaded subject data:")
    print(f"  Epochs shape: {data['epochs'].shape}")
    print(f"  Stages shape: {data['stages'].shape}")
    print(f"  Channels: {data['channel_names']}")
    print(f"  Sampling rate: {data['sampling_rate']} Hz")
    print(f"  Stage distribution:")
    
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        count = np.sum(data['stages'] == stage_idx)
        print(f"    {stage_name}: {count} epochs ({count * 30 / 60:.1f} min)")


if __name__ == "__main__":
    main()
