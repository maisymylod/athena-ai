"""Evaluate the trained classifier's ROC-AUC under common image-quality
perturbations (JPEG recompression at q=50, 2x downsample/upsample, 5°
rotation). Reports clean-vs-perturbed AUROC so we can be honest about
robustness in EVAL.md.

Usage:
    python scripts/eval_robustness.py \
        --checkpoint checkpoints/best_model.pt \
        --data-dir data/raw \
        --output robustness.json
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Allow running this as a script ("python scripts/eval_robustness.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.config import TrainConfig
from ml.dataset import build_eval_transforms, load_image_paths, stratified_split
from ml.evaluate import compute_metrics
from ml.model import DeepfakeDetector
from ml.train import get_device

logger = logging.getLogger(__name__)


class PerturbedDataset(Dataset):
    """Wrap a list of (path, label) pairs and apply a perturbation pipeline
    on top of the standard eval transforms."""

    def __init__(self, paths, labels, perturbation, image_size=224):
        self.paths = paths
        self.labels = labels
        self.perturbation = perturbation
        self.normalize = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.perturbation(img)
        return self.normalize(img), torch.tensor(self.labels[idx], dtype=torch.float32)


def jpeg_q(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downsample(img, factor=2):
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def rotate(img, deg):
    return img.rotate(deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", default=32, type=int)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = get_device()
    logger.info("device: %s", device)

    paths, labels = load_image_paths(args.data_dir)
    config = TrainConfig(data_dir=args.data_dir)
    _, _, (test_paths, test_labels) = stratified_split(paths, labels, config)
    logger.info("test split: %d images", len(test_paths))

    model = DeepfakeDetector(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    perturbations = {
        "clean": lambda img: img,
        "jpeg_q50": lambda img: jpeg_q(img, 50),
        "jpeg_q30": lambda img: jpeg_q(img, 30),
        "downsample_2x": lambda img: downsample(img, 2),
        "rotate_5deg": lambda img: rotate(img, 5),
    }

    results = {}
    for name, fn in perturbations.items():
        ds = PerturbedDataset(test_paths, test_labels, fn)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        m = compute_metrics(model, loader, device)
        logger.info("%s: AUROC=%.4f acc=%.4f f1=%.4f", name, m["roc_auc"], m["accuracy"], m["f1"])
        results[name] = m

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
