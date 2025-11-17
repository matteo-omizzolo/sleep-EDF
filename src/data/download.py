"""
Download Sleep-EDF dataset from PhysioNet.

References:
- https://physionet.org/content/sleep-edfx/1.0.0/
"""

import os
import requests
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm


PHYSIONET_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"


def download_file(url: str, output_path: Path, overwrite: bool = False) -> bool:
    """Download a single file with progress bar."""
    if output_path.exists() and not overwrite:
        print(f"Skipping {output_path.name} (already exists)")
        return False
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def get_subject_files(subject_id: str, night: int = 0) -> List[tuple]:
    """
    Get file URLs for a subject.
    
    Args:
        subject_id: Subject ID (e.g., "SC4001")
        night: Night number (0 or 1)
    
    Returns:
        List of (filename, url) tuples
    """
    night_suffix = f"{night:01d}" if night < 10 else str(night)
    
    # Sleep-EDF Expanded has two subdirectories
    if subject_id.startswith("SC"):
        subdir = "sleep-cassette/"
    else:
        subdir = "sleep-telemetry/"
    
    base_name = f"{subject_id}{night_suffix}"
    
    files = [
        (f"{base_name}-PSG.edf", f"{PHYSIONET_BASE_URL}{subdir}{base_name}-PSG.edf"),
        (f"{base_name}-Hypnogram.edf", f"{PHYSIONET_BASE_URL}{subdir}{base_name}-Hypnogram.edf"),
    ]
    
    return files


def download_subject(
    subject_id: str,
    output_dir: Path,
    nights: List[int] = [0],
    overwrite: bool = False
) -> bool:
    """
    Download all files for a subject.
    
    Args:
        subject_id: Subject ID (e.g., "SC4001")
        output_dir: Output directory
        nights: List of night numbers to download
        overwrite: Overwrite existing files
    
    Returns:
        True if successful
    """
    print(f"\nDownloading {subject_id}...")
    
    success = True
    for night in nights:
        files = get_subject_files(subject_id, night)
        
        for filename, url in files:
            output_path = output_dir / filename
            downloaded = download_file(url, output_path, overwrite)
            
            if not downloaded and not output_path.exists():
                success = False
    
    return success


def download_sleep_edf_expanded(
    output_dir: str,
    n_subjects: Optional[int] = None,
    cohort: str = "cassette",
    nights_per_subject: int = 1,
    overwrite: bool = False
) -> None:
    """
    Download Sleep-EDF Expanded dataset.
    
    Args:
        output_dir: Output directory for downloaded files
        n_subjects: Number of subjects to download (None = all)
        cohort: "cassette" (healthy) or "telemetry" (sleep disorder patients)
        nights_per_subject: Number of nights to download per subject (1 or 2)
        overwrite: Overwrite existing files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sleep-Cassette subjects (healthy, 2 nights each)
    cassette_subjects = [f"SC4{i:02d}1" for i in range(1, 21)]  # SC4001-SC4201
    
    # Sleep-Telemetry subjects (sleep disorder patients, 2 nights each)
    telemetry_subjects = [f"ST7{i:02d}1" for i in range(1, 23)]  # ST7011-ST7221
    
    if cohort == "cassette":
        subjects = cassette_subjects
    elif cohort == "telemetry":
        subjects = telemetry_subjects
    else:
        subjects = cassette_subjects + telemetry_subjects
    
    if n_subjects is not None:
        subjects = subjects[:n_subjects]
    
    print(f"Downloading {len(subjects)} subjects from {cohort} cohort...")
    print(f"Output directory: {output_path}")
    
    nights = list(range(nights_per_subject))
    
    failed = []
    for subject_id in subjects:
        success = download_subject(subject_id, output_path, nights, overwrite)
        if not success:
            failed.append(subject_id)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Downloaded: {len(subjects) - len(failed)}/{len(subjects)} subjects")
    
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Sleep-EDF Expanded dataset from PhysioNet"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of subjects to download (default: all)"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=["cassette", "telemetry", "both"],
        default="cassette",
        help="Which cohort to download (default: cassette)"
    )
    parser.add_argument(
        "--nights",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of nights per subject (default: 1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    download_sleep_edf_expanded(
        output_dir=args.output,
        n_subjects=args.n_subjects,
        cohort=args.cohort,
        nights_per_subject=args.nights,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
