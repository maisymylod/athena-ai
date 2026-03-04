"""Tests for DeepfakeDetector model architecture."""

import torch
import pytest

from ml.model import DeepfakeDetector


class TestDeepfakeDetector:
    def test_output_shape(self, model: DeepfakeDetector):
        """Model should output a single logit per sample."""
        batch = torch.randn(4, 3, 224, 224)
        output = model(batch)
        assert output.shape == (4,)

    def test_output_shape_single(self, model: DeepfakeDetector):
        """Single image should also produce correct shape."""
        batch = torch.randn(1, 3, 224, 224)
        output = model(batch)
        assert output.shape == (1,)

    def test_freeze_backbone(self, model: DeepfakeDetector):
        """Freezing backbone should disable gradients on feature layers."""
        model.freeze_backbone()
        for param in model.features.parameters():
            assert not param.requires_grad
        # Classifier should still be trainable
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model: DeepfakeDetector):
        """Unfreezing should re-enable gradients on feature layers."""
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.features.parameters():
            assert param.requires_grad

    def test_forward_backward(self, model: DeepfakeDetector):
        """Model should support full forward-backward pass."""
        batch = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0.0, 1.0])

        output = model(batch)
        loss = torch.nn.BCEWithLogitsLoss()(output, labels)
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_different_input_sizes(self, model: DeepfakeDetector):
        """Model should handle different input resolutions via adaptive pooling."""
        for size in [64, 128, 224, 299]:
            batch = torch.randn(1, 3, size, size)
            output = model(batch)
            assert output.shape == (1,)
