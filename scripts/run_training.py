"""CLI entry point for the full Athena training pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import TrainConfig
from ml.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Athena — Train deepfake detection model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to data directory with real/ and synthetic/ subdirs (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Path to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Epochs to freeze backbone (default: 3)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )

    config = TrainConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print("=" * 60)
    print("  ATHENA — Deepfake Detection Model Training")
    print("=" * 60)
    print(f"  Data:       {config.data_dir}")
    print(f"  Output:     {config.output_dir}")
    print(f"  Epochs:     {config.epochs} (freeze: {config.freeze_epochs})")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print(f"  Patience:   {config.patience}")
    print(f"  Seed:       {config.seed}")
    print("=" * 60)
    print()

    trainer = Trainer(config)
    results = trainer.train()

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Best val loss: {results['best_val_loss']:.4f}")
    print(f"  Checkpoint:    {results['checkpoint_path']}")
    print()
    print("  Test Metrics:")
    for key, value in results["test_metrics"].items():
        if key != "confusion_matrix":
            print(f"    {key}: {value}")
    cm = results["test_metrics"]["confusion_matrix"]
    print(f"    Confusion matrix: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
