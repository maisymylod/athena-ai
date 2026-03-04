"""DeepfakeDetector — EfficientNet-B0 with custom classification head."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeDetector(nn.Module):
    """Binary classifier for detecting AI-generated images.

    Architecture:
        - Backbone: EfficientNet-B0 pretrained on ImageNet
        - Head: AdaptiveAvgPool -> Dropout -> Linear(1280, 512) -> ReLU
                -> Dropout -> Linear(512, 1)
        - Output: single logit (use BCEWithLogitsLoss)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Keep everything except the original classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(-1)

    def freeze_backbone(self) -> None:
        """Freeze the EfficientNet backbone for transfer learning."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
