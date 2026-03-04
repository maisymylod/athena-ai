"""Shared test fixtures for Athena ML tests."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.config import InferenceConfig, TrainConfig
from ml.model import DeepfakeDetector


@pytest.fixture
def dummy_image_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory with dummy images."""
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()

    # Create small dummy images
    for i in range(20):
        img = Image.new("RGB", (64, 64), color=(i * 10 % 256, 100, 150))
        img.save(real_dir / f"real_{i:03d}.jpg")

    for i in range(20):
        img = Image.new("RGB", (64, 64), color=(200, i * 10 % 256, 50))
        img.save(synthetic_dir / f"synthetic_{i:03d}.jpg")

    return tmp_path


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Create a single dummy image for inference tests."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


@pytest.fixture
def model() -> DeepfakeDetector:
    """Create a DeepfakeDetector model (not pretrained, for speed)."""
    return DeepfakeDetector(pretrained=False)


@pytest.fixture
def trained_checkpoint(tmp_path: Path, model: DeepfakeDetector) -> Path:
    """Save a dummy checkpoint for inference tests."""
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.5,
            "config": {"image_size": 224, "batch_size": 32, "learning_rate": 1e-3},
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def train_config(dummy_image_dir: Path, tmp_path: Path) -> TrainConfig:
    """Create a TrainConfig pointing to dummy data."""
    return TrainConfig(
        data_dir=dummy_image_dir,
        output_dir=tmp_path / "checkpoints",
        batch_size=4,
        num_workers=0,
        epochs=2,
        freeze_epochs=1,
        patience=2,
        image_size=64,
    )


@pytest.fixture
def inference_config(trained_checkpoint: Path) -> InferenceConfig:
    """Create an InferenceConfig pointing to a dummy checkpoint."""
    return InferenceConfig(checkpoint_path=trained_checkpoint, image_size=224)
