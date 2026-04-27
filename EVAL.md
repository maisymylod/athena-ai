# Athena — Detector Evaluation

This document records how the synthetic-image classifier shipped in
`ml/model.py` was trained and how it performs on held-out data.

> **Status:** placeholder — fill in before publishing the demo.
> Numbers below are the *target* numbers we are training toward, not
> measured results. Replace each `TODO` once the training run completes
> and `ml/evaluate.py` has been run on the held-out test split.

## Architecture

- **Backbone:** EfficientNet-B0, pretrained on ImageNet
  (`torchvision.models.efficientnet_b0`).
- **Head:** AdaptiveAvgPool → Dropout(0.3) → Linear(1280, 512) → ReLU →
  Dropout(0.2) → Linear(512, 1).
- **Output:** single logit, trained with `BCEWithLogitsLoss`.
- **Two-phase fine-tune:** phase 1 freezes the backbone and trains only the
  classifier head; phase 2 unfreezes the backbone at 1/10 the learning rate.

See `ml/model.py:DeepfakeDetector` and `ml/train.py:Trainer`.

## Training data

| Source | Real | Synthetic | Generators | Resolution |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

Splits (stratified by label, seed=42, configured in
`ml/config.py:TrainConfig`):

- Train: 75%
- Val: 15%
- Test: 10%

## Augmentations

Configured in `ml/dataset.py` and `ml/config.py:AugmentConfig`.

- Resize to 224×224
- RandomHorizontalFlip(p=0.5)
- RandomRotation(±10°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- GaussianBlur, kernel=5, p=0.2
- JPEGCompressionAugment(quality 30–95) — simulates real-world recompression
- Normalize with ImageNet stats
- RandomErasing(p=0.3)

## Headline metrics

Computed by `ml/evaluate.py:compute_metrics` on the held-out test split.

| Metric | Value |
|---|---|
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| ROC-AUC | TODO |

Confusion matrix:

|  | Pred Real | Pred Synthetic |
|---|---|---|
| **Real** | TODO TN | TODO FP |
| **Synthetic** | TODO FN | TODO TP |

## Robustness

The same test split, run through additional perturbations to simulate the
distortions content typically picks up after being uploaded, screenshotted,
or recompressed.

| Perturbation | ROC-AUC |
|---|---|
| Clean | TODO |
| JPEG quality 50 | TODO |
| 2× downsample then upsample | TODO |
| 5° rotation | TODO |

## Cross-generator generalization

Held-out generators (trained on N−1, tested on the held-out one):

| Held-out generator | ROC-AUC |
|---|---|
| TODO | TODO |

Cross-generator generalization is the hardest robustness axis for any
synthetic-image detector. We expect noticeable degradation here — be
honest in the model card about which generators we have not seen.

## Limitations

- We have not evaluated the model on adversarially-perturbed inputs. A
  motivated attacker can almost certainly bypass it; the operating
  assumption is that most uploaded deepfakes today are not adversarial.
- The classifier only inspects pixels. We do not currently use C2PA
  provenance, watermark detection, or video-level cues.
- This is **not** a substitute for expert review when the result will be
  used as evidence in a legal proceeding.

## How to reproduce

```bash
# 1. Get the dataset
python ml/download_dataset.py --list
# follow the printed instructions, then:
mkdir -p data/raw/real data/raw/synthetic
# (copy images into the appropriate subdirectories)

# 2. Train
python scripts/run_training.py --data-dir data/raw --output-dir checkpoints

# 3. Evaluate (writes JSON to stdout or --output)
python -m ml.evaluate --checkpoint checkpoints/best_model.pt \
    --data-dir data/raw --output eval_metrics.json
```
