# Athena — Detector Evaluation

This document records how the synthetic-image classifier shipped at
`checkpoints/best_model.pt` was trained and how it performs on held-out
data.

## Architecture

- **Backbone:** EfficientNet-B0, pretrained on ImageNet
  (`torchvision.models.efficientnet_b0`).
- **Head:** AdaptiveAvgPool → Dropout(0.3) → Linear(1280, 512) → ReLU →
  Dropout(0.2) → Linear(512, 1).
- **Output:** single logit, trained with `BCEWithLogitsLoss`.
- **Two-phase fine-tune:** phase 1 (3 epochs) freezes the backbone and
  trains only the classifier head; phase 2 (9 epochs) unfreezes the
  backbone at 1/10 the learning rate.

See `ml/model.py:DeepfakeDetector` and `ml/train.py:Trainer`.

## Training data

Pulled from
[Madushan996/real_fake_images](https://huggingface.co/datasets/Madushan996/real_fake_images)
on the Hugging Face Hub via `scripts/prepare_dataset.py`.

| Source | Real | Synthetic | Generators | Resolution |
|---|---|---|---|---|
| Madushan996/real_fake_images (subset) | 1197 | 1197 | Flux + Stable Diffusion 1.5 + one more | mostly 512×512 |

Splits (stratified by label, seed=42, configured in
`ml/config.py:TrainConfig`):

- Train: 1794 images
- Val: 358 images
- Test: 242 images

Six images (3 per class) are held out before the split and used as the
landing-page demo's "Try with these" examples — they are not part of
train, val, or test.

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

## Training run

12 epochs total (3 frozen-backbone + 9 fine-tuned), batch size 32,
trained on Apple Silicon MPS in **~4 minutes**. Best checkpoint from the
last epoch (val_loss 0.0021, val_acc 100%).

```
Epoch  1/3  [14.0s]  train_acc=0.8974  val_loss=0.0673  val_acc=0.9860
Epoch  2/3  [10.6s]  train_acc=0.9443  val_loss=0.0558  val_acc=0.9888
Epoch  3/3  [10.6s]  train_acc=0.9359  val_loss=0.0738  val_acc=0.9777
Epoch  4/12 [28.6s]  train_acc=0.9666  val_loss=0.0354  val_acc=0.9832
Epoch  5/12 [25.8s]  train_acc=0.9788  val_loss=0.0153  val_acc=0.9916
Epoch  6/12 [26.0s]  train_acc=0.9827  val_loss=0.0126  val_acc=0.9972
Epoch  7/12 [25.9s]  train_acc=0.9933  val_loss=0.0144  val_acc=0.9944
Epoch  8/12 [26.1s]  train_acc=0.9900  val_loss=0.0159  val_acc=0.9944
Epoch  9/12 [25.4s]  train_acc=0.9916  val_loss=0.0062  val_acc=1.0000
Epoch 10/12 [25.4s]  train_acc=0.9928  val_loss=0.0119  val_acc=0.9944
Epoch 11/12 [24.8s]  train_acc=0.9916  val_loss=0.0102  val_acc=0.9944
Epoch 12/12 [24.7s]  train_acc=0.9967  val_loss=0.0021  val_acc=1.0000
```

## Headline metrics

Computed by `ml/evaluate.py:compute_metrics` on the held-out test split
(242 images, 121 real / 121 synthetic).

| Metric | Value |
|---|---|
| Accuracy | 0.9959 |
| Precision | 0.9918 |
| Recall | 1.0000 |
| F1 | 0.9959 |
| **ROC-AUC** | **0.9999** |

Confusion matrix:

|  | Pred Real | Pred Synthetic |
|---|---|---|
| **Real** | 120 (TN) | 1 (FP) |
| **Synthetic** | 0 (FN) | 121 (TP) |

One false positive, zero false negatives. The model is currently more
willing to flag a real image than to miss a synthetic one — a
deliberate bias for this domain (false negatives are worse: a missed
deepfake is content that stays online).

## Robustness

Same test split, run through additional perturbations to simulate the
distortions content typically picks up after being uploaded,
screenshotted, or recompressed. Reproduce with
`python scripts/eval_robustness.py --checkpoint checkpoints/best_model.pt --data-dir data/raw`.

| Perturbation | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Clean | 0.9959 | 0.9959 | **0.9999** |
| JPEG quality 50 | 0.9876 | 0.9877 | 0.9999 |
| JPEG quality 30 | 0.9917 | 0.9918 | 0.9999 |
| 2× downsample then upsample | 0.9959 | 0.9959 | 0.9999 |
| 5° rotation | 1.0000 | 1.0000 | 1.0000 |

ROC-AUC stays at ≥0.9999 across all perturbations, which suggests the
model's decision boundary is robust even when accuracy at the
0.5 threshold drifts slightly under heavy JPEG compression.

## Limitations

- **Single dataset, single domain.** All training and test images come
  from the same Hugging Face dataset family. The high in-distribution
  numbers above do **not** mean the model will perform this well on
  arbitrary internet imagery, on faces specifically, or on generators we
  have not seen during training. Cross-dataset evaluation is the next
  open task; we expect noticeable degradation.
- **No adversarial robustness.** A motivated attacker can almost
  certainly bypass the classifier with pixel-level perturbations. The
  operating assumption is that most uploaded deepfakes today are not
  adversarial.
- **Not face-specific.** This first checkpoint is a general
  real-vs-synthetic classifier. A face-specific fine-tune is on the
  roadmap once we have access to the 140k Real-Fake-Faces dataset
  (Kaggle, requires auth).
- **Pixels only.** We do not currently use C2PA provenance, watermark
  detection, or video-level cues.
- **Not a substitute for expert review** when the result will be used
  as evidence in a legal proceeding.

## How to reproduce

```bash
# 1. Download dataset + lay it out under data/raw/
python scripts/prepare_dataset.py

# 2. Train (Apple MPS, CUDA, or CPU — auto-detected)
python scripts/run_training.py \
    --data-dir data/raw --output-dir checkpoints \
    --epochs 12 --freeze-epochs 3 --batch-size 32 --num-workers 0

# 3. Headline metrics on the test split
python -m ml.evaluate                         # not yet wrapped as a CLI; see ml/evaluate.py

# 4. Robustness sweep
python scripts/eval_robustness.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/raw \
    --output robustness.json
```
