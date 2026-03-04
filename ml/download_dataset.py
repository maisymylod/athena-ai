"""Script to download public AI vs real image datasets for training.

Provides instructions and helper functions for downloading datasets
suitable for deepfake detection training.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS = {
    "cifake": {
        "name": "CIFAKE: Real and AI-Generated Synthetic Images",
        "url": "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images",
        "description": "60K real images (CIFAR-10) + 60K AI-generated images (Stable Diffusion). 32x32 resolution.",
        "command": "kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images",
    },
    "140k-faces": {
        "name": "140K Real and Fake Faces",
        "url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
        "description": "70K real face photos (Flickr) + 70K StyleGAN-generated faces. 256x256 resolution.",
        "command": "kaggle datasets download -d xhlulu/140k-real-and-fake-faces",
    },
    "deepfake-faces": {
        "name": "Deepfake and Real Images",
        "url": "https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images",
        "description": "Mixed dataset of real and deepfake-generated face images.",
        "command": "kaggle datasets download -d manjilkarki/deepfake-and-real-images",
    },
}


def print_instructions(dataset_key: str | None = None) -> None:
    """Print download instructions for available datasets."""
    print("=" * 60)
    print("  ATHENA — Dataset Download Instructions")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. Install the Kaggle CLI: pip install kaggle")
    print("  2. Set up your Kaggle API token:")
    print("     - Go to https://www.kaggle.com/settings")
    print("     - Click 'Create New Token'")
    print("     - Save kaggle.json to ~/.kaggle/kaggle.json")
    print()

    datasets = {dataset_key: DATASETS[dataset_key]} if dataset_key else DATASETS

    for key, info in datasets.items():
        print(f"  [{key}] {info['name']}")
        print(f"  {info['description']}")
        print(f"  URL: {info['url']}")
        print(f"  Download: {info['command']}")
        print()

    print("After downloading, organize files into:")
    print("  data/raw/real/      — real images")
    print("  data/raw/synthetic/ — AI-generated images")
    print()
    print("Example:")
    print("  mkdir -p data/raw/real data/raw/synthetic")
    print("  # Move/copy images into the appropriate directories")
    print("=" * 60)


def setup_data_dirs(base_dir: Path) -> None:
    """Create the expected data directory structure."""
    (base_dir / "real").mkdir(parents=True, exist_ok=True)
    (base_dir / "synthetic").mkdir(parents=True, exist_ok=True)
    print(f"Created data directories at {base_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Download and set up datasets for Athena deepfake detection"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Specific dataset to show instructions for",
    )
    parser.add_argument(
        "--setup-dirs",
        type=str,
        default="data/raw",
        help="Create data directory structure (default: data/raw)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )

    args = parser.parse_args()

    if args.list or not any([args.dataset]):
        print_instructions()
    elif args.dataset:
        print_instructions(args.dataset)

    if args.setup_dirs:
        setup_data_dirs(Path(args.setup_dirs))


if __name__ == "__main__":
    main()
