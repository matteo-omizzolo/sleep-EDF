#!/usr/bin/env python3
"""
Download Sleep-EDF Expanded dataset from PhysioNet.

Usage:
    python scripts/download_data.py --output data/raw --n-subjects 30
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.download import download_sleep_edf_expanded


def main():
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
        default=30,
        help="Number of subjects to download (default: 30)"
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
