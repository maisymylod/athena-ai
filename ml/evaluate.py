"""Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics on a dataset.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, and confusion_matrix.
    """
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(float)

    # Confusion matrix components
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # ROC-AUC (manual implementation to avoid sklearn dependency at eval time)
    roc_auc = _compute_roc_auc(y_true, y_prob)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "total_samples": len(y_true),
    }


def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC score using the trapezoidal rule."""
    if len(np.unique(y_true)) < 2:
        return 0.0

    # Sort by descending probability
    desc_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp_count = 0.0
    fp_count = 0.0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for label in y_true_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1

        tpr = tp_count / n_pos
        fpr = fp_count / n_neg

        # Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2

        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)
