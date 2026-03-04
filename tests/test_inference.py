"""Tests for the DeepfakeInference pipeline."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.inference import DeepfakeInference
from ml.model import DeepfakeDetector


class TestDeepfakeInference:
    def test_from_checkpoint(self, trained_checkpoint: Path):
        inference = DeepfakeInference.from_checkpoint(trained_checkpoint)
        assert inference.model is not None
        assert inference.device is not None

    def test_predict_returns_tuple(self, trained_checkpoint: Path, dummy_image: Path):
        inference = DeepfakeInference.from_checkpoint(trained_checkpoint)
        result = inference.predict(dummy_image)

        assert isinstance(result, tuple)
        assert len(result) == 3

        is_synthetic, confidence, indicators = result
        assert isinstance(is_synthetic, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(indicators, list)

    def test_predict_pil(self, trained_checkpoint: Path):
        inference = DeepfakeInference.from_checkpoint(trained_checkpoint)
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))

        is_synthetic, confidence, indicators = inference.predict_pil(img)
        assert isinstance(is_synthetic, bool)
        assert 0.0 <= confidence <= 1.0

    def test_predict_pil_converts_mode(self, trained_checkpoint: Path):
        """Should handle non-RGB images (e.g., RGBA, L)."""
        inference = DeepfakeInference.from_checkpoint(trained_checkpoint)
        img = Image.new("RGBA", (224, 224), color=(100, 100, 100, 255))

        is_synthetic, confidence, indicators = inference.predict_pil(img)
        assert isinstance(is_synthetic, bool)

    def test_indicators_populated(self, trained_checkpoint: Path, dummy_image: Path):
        inference = DeepfakeInference.from_checkpoint(trained_checkpoint)
        _, _, indicators = inference.predict(dummy_image)
        assert len(indicators) >= 1

    def test_missing_checkpoint_raises(self, tmp_path: Path):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            DeepfakeInference.from_checkpoint(tmp_path / "nonexistent.pt")
