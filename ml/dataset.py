"""Dataset loading, augmentation, and stratified splitting for deepfake detection."""

import io
import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import AugmentConfig, TrainConfig

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class JPEGCompressionAugment:
    """Simulate JPEG compression artifacts — common in real-world deepfakes."""

    def __init__(self, quality_range: tuple[int, int] = (30, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_train_transforms(
    image_size: int = 224, aug_config: AugmentConfig | None = None
) -> transforms.Compose:
    """Build training augmentation pipeline."""
    cfg = aug_config or AugmentConfig()
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_p),
        transforms.RandomRotation(cfg.rotation_degrees),
        transforms.ColorJitter(
            brightness=cfg.color_jitter_brightness,
            contrast=cfg.color_jitter_contrast,
            saturation=cfg.color_jitter_saturation,
            hue=cfg.color_jitter_hue,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5)], p=cfg.gaussian_blur_p
        ),
        transforms.Lambda(
            lambda img: JPEGCompressionAugment(cfg.jpeg_quality_range)(img)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=cfg.random_erasing_p),
    ])


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build evaluation/inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DeepfakeDataset(Dataset):
    """Dataset for real vs synthetic image classification.

    Expects directory structure:
        data_dir/real/     — real images (label 0)
        data_dir/synthetic/ — AI-generated images (label 1)
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = float(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


def load_image_paths(data_dir: Path) -> tuple[list[Path], list[int]]:
    """Load image paths and labels from directory structure.

    Returns:
        Tuple of (paths, labels) where label 0=real, 1=synthetic.
    """
    paths: list[Path] = []
    labels: list[int] = []

    real_dir = data_dir / "real"
    synthetic_dir = data_dir / "synthetic"

    for directory, label in [(real_dir, 0), (synthetic_dir, 1)]:
        if not directory.exists():
            logger.warning("Directory not found: %s", directory)
            continue

        count = 0
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(f)
                labels.append(label)
                count += 1

        logger.info("Loaded %d images from %s (label=%d)", count, directory, label)

    return paths, labels


def stratified_split(
    paths: list[Path], labels: list[int], config: TrainConfig
) -> tuple[
    tuple[list[Path], list[int]],
    tuple[list[Path], list[int]],
    tuple[list[Path], list[int]],
]:
    """Perform stratified train/val/test split.

    Returns:
        ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    random.seed(config.seed)

    # Group by label
    label_groups: dict[int, list[Path]] = {}
    for path, label in zip(paths, labels):
        label_groups.setdefault(label, []).append(path)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for label, group_paths in label_groups.items():
        shuffled = list(group_paths)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * config.train_split)
        n_val = int(n * config.val_split)

        train_paths.extend(shuffled[:n_train])
        train_labels.extend([label] * n_train)

        val_paths.extend(shuffled[n_train : n_train + n_val])
        val_labels.extend([label] * (len(shuffled[n_train : n_train + n_val])))

        test_paths.extend(shuffled[n_train + n_val :])
        test_labels.extend([label] * (len(shuffled[n_train + n_val :])))

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )
