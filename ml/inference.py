"""DeepfakeInference — load model checkpoint and predict on single images."""

import logging
from pathlib import Path

import torch
from PIL import Image

from .config import InferenceConfig
from .dataset import build_eval_transforms
from .model import DeepfakeDetector

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Auto-detect best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeepfakeInference:
    """Load a trained deepfake detection model and run inference on images.

    Usage:
        inference = DeepfakeInference.from_checkpoint("checkpoints/best_model.pt")
        result = inference.predict("suspect_image.jpg")
        print(result)  # (True, 0.94, ["High synthetic confidence: 94%"])
    """

    def __init__(self, model: DeepfakeDetector, device: torch.device, config: InferenceConfig):
        self.model = model
        self.device = device
        self.config = config
        self.transform = build_eval_transforms(config.image_size)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: InferenceConfig | None = None,
    ) -> "DeepfakeInference":
        """Load model from a training checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            config: Optional inference configuration.

        Returns:
            Initialized DeepfakeInference instance.
        """
        config = config or InferenceConfig(checkpoint_path=Path(checkpoint_path))
        device = _get_device()

        model = DeepfakeDetector(pretrained=False)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss=%.4f) on %s",
            checkpoint_path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", -1),
            device,
        )

        return cls(model, device, config)

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> tuple[bool, float, list[str]]:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (is_synthetic, confidence, indicators).
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        logit = self.model(tensor)
        confidence = torch.sigmoid(logit).item()

        is_synthetic = confidence >= self.config.confidence_threshold

        indicators = []
        if is_synthetic:
            indicators.append(f"ML model confidence: {confidence:.0%}")
            if confidence >= 0.9:
                indicators.append("Very high synthetic probability")
            elif confidence >= 0.7:
                indicators.append("High synthetic probability")
        else:
            indicators.append(f"ML model confidence (real): {1 - confidence:.0%}")

        return is_synthetic, confidence, indicators

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> tuple[bool, float, list[str]]:
        """Run inference on a PIL Image object.

        Args:
            img: PIL Image in any mode (will be converted to RGB).

        Returns:
            Tuple of (is_synthetic, confidence, indicators).
        """
        img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        logit = self.model(tensor)
        confidence = torch.sigmoid(logit).item()

        is_synthetic = confidence >= self.config.confidence_threshold

        indicators = []
        if is_synthetic:
            indicators.append(f"ML model confidence: {confidence:.0%}")
            if confidence >= 0.9:
                indicators.append("Very high synthetic probability")
            elif confidence >= 0.7:
                indicators.append("High synthetic probability")
        else:
            indicators.append(f"ML model confidence (real): {1 - confidence:.0%}")

        return is_synthetic, confidence, indicators
