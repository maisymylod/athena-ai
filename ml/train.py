"""Trainer class: training loop, early stopping, checkpointing."""

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import (
    DeepfakeDataset,
    build_eval_transforms,
    build_train_transforms,
    load_image_paths,
    stratified_split,
)
from .evaluate import compute_metrics
from .model import DeepfakeDetector

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Handles the full training pipeline with early stopping and checkpointing."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = get_device()

        set_seed(config.seed)
        logger.info("Using device: %s", self.device)

        self.model = DeepfakeDetector(pretrained=True).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        config.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test dataloaders."""
        paths, labels = load_image_paths(self.config.data_dir)
        if not paths:
            raise FileNotFoundError(
                f"No images found in {self.config.data_dir}. "
                "Expected subdirectories: real/ and synthetic/"
            )

        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
            stratified_split(paths, labels, self.config)
        )

        train_ds = DeepfakeDataset(
            train_paths, train_labels, build_train_transforms(self.config.image_size)
        )
        val_ds = DeepfakeDataset(
            val_paths, val_labels, build_eval_transforms(self.config.image_size)
        )
        test_ds = DeepfakeDataset(
            test_paths, test_labels, build_eval_transforms(self.config.image_size)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def _train_epoch(
        self, loader: DataLoader, optimizer: AdamW
    ) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        """Validate the model. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Save model checkpoint."""
        path = self.config.output_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "image_size": self.config.image_size,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                },
            },
            path,
        )
        logger.info("Saved checkpoint to %s (val_loss=%.4f)", path, val_loss)
        return path

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.config.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                self.patience_counter,
            )
            return True
        return False

    def train(self) -> dict:
        """Run the full training pipeline.

        Two-phase training:
            Phase 1: Freeze backbone, train only classifier head
            Phase 2: Unfreeze backbone, fine-tune entire network

        Returns:
            Dict with training results and test metrics.
        """
        logger.info("Starting training with config: %s", self.config)
        train_loader, val_loader, test_loader = self._build_dataloaders()

        # Phase 1: Frozen backbone
        self.model.freeze_backbone()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        logger.info("Phase 1: Training classifier head (%d epochs)", self.config.freeze_epochs)
        for epoch in range(self.config.freeze_epochs):
            start = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            val_loss, val_acc = self._validate(val_loader)
            scheduler.step(val_loss)
            elapsed = time.time() - start

            logger.info(
                "Epoch %d/%d [%.1fs] — train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1,
                self.config.freeze_epochs,
                elapsed,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

        # Phase 2: Fine-tune entire network
        self.model.unfreeze_backbone()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        self.patience_counter = 0

        remaining_epochs = self.config.epochs - self.config.freeze_epochs
        logger.info("Phase 2: Fine-tuning full network (%d epochs)", remaining_epochs)

        for epoch in range(remaining_epochs):
            start = time.time()
            global_epoch = epoch + self.config.freeze_epochs
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            val_loss, val_acc = self._validate(val_loader)
            scheduler.step(val_loss)
            elapsed = time.time() - start

            logger.info(
                "Epoch %d/%d [%.1fs] — train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                global_epoch + 1,
                self.config.epochs,
                elapsed,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_loss < self.best_val_loss:
                self._save_checkpoint(global_epoch, val_loss)

            if self._check_early_stopping(val_loss):
                break

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        checkpoint = torch.load(
            self.config.output_dir / "best_model.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = compute_metrics(self.model, test_loader, self.device)
        logger.info("Test metrics: %s", test_metrics)

        return {
            "best_val_loss": self.best_val_loss,
            "checkpoint_path": str(self.config.output_dir / "best_model.pt"),
            "test_metrics": test_metrics,
        }
