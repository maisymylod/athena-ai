"""Configuration dataclasses for the Athena ML pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("checkpoints")
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    freeze_epochs: int = 3
    patience: int = 5
    image_size: int = 224
    train_split: float = 0.75
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration."""

    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    image_size: int = 224
    confidence_threshold: float = 0.5


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation configuration for deepfake detection training."""

    jpeg_quality_range: tuple[int, int] = (30, 95)
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    random_erasing_p: float = 0.3
    horizontal_flip_p: float = 0.5
    rotation_degrees: int = 10
    gaussian_blur_p: float = 0.2
