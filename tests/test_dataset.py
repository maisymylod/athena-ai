"""Tests for dataset loading, augmentation, and splitting."""

from pathlib import Path

import torch
import pytest

from ml.config import TrainConfig
from ml.dataset import (
    DeepfakeDataset,
    build_eval_transforms,
    build_train_transforms,
    load_image_paths,
    stratified_split,
)


class TestLoadImagePaths:
    def test_loads_both_classes(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        assert len(paths) == 40  # 20 real + 20 synthetic
        assert labels.count(0) == 20
        assert labels.count(1) == 20

    def test_handles_missing_directory(self, tmp_path: Path):
        paths, labels = load_image_paths(tmp_path / "nonexistent")
        assert len(paths) == 0
        assert len(labels) == 0

    def test_ignores_non_image_files(self, dummy_image_dir: Path):
        # Create a non-image file
        (dummy_image_dir / "real" / "readme.txt").write_text("not an image")
        paths, labels = load_image_paths(dummy_image_dir)
        assert len(paths) == 40  # Still 40, text file ignored


class TestStratifiedSplit:
    def test_split_ratios(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        config = TrainConfig(train_split=0.75, val_split=0.15, test_split=0.10)
        (train_p, train_l), (val_p, val_l), (test_p, test_l) = stratified_split(
            paths, labels, config
        )
        total = len(train_p) + len(val_p) + len(test_p)
        assert total == 40

    def test_stratification_preserves_labels(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        config = TrainConfig(train_split=0.75, val_split=0.15, test_split=0.10)
        (_, train_l), (_, val_l), (_, test_l) = stratified_split(
            paths, labels, config
        )
        # Each split should contain both classes
        assert 0 in train_l and 1 in train_l
        assert 0 in val_l and 1 in val_l

    def test_reproducibility(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        config = TrainConfig(seed=42)
        split1 = stratified_split(paths, labels, config)
        split2 = stratified_split(paths, labels, config)
        assert split1[0][0] == split2[0][0]  # Same train paths


class TestDeepfakeDataset:
    def test_getitem_returns_tensor_and_label(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        transform = build_eval_transforms(image_size=64)
        ds = DeepfakeDataset(paths, labels, transform)

        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, float)

    def test_length(self, dummy_image_dir: Path):
        paths, labels = load_image_paths(dummy_image_dir)
        ds = DeepfakeDataset(paths, labels)
        assert len(ds) == 40


class TestTransforms:
    def test_train_transform_output_shape(self, dummy_image_dir: Path):
        transform = build_train_transforms(image_size=64)
        paths, labels = load_image_paths(dummy_image_dir)
        ds = DeepfakeDataset(paths, labels, transform)
        img, _ = ds[0]
        assert img.shape == (3, 64, 64)

    def test_eval_transform_output_shape(self, dummy_image_dir: Path):
        transform = build_eval_transforms(image_size=128)
        paths, labels = load_image_paths(dummy_image_dir)
        ds = DeepfakeDataset(paths, labels, transform)
        img, _ = ds[0]
        assert img.shape == (3, 128, 128)
