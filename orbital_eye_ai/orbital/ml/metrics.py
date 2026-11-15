"""Segmentation evaluation metrics (NumPy only, usable in Colab and in tests)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def per_class_scores(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, List[Optional[float]]]:
    """Dice, IoU, precision and recall for every class. None when a class is absent from both."""
    dice, iou, precision, recall = [], [], [], []
    for c in range(num_classes):
        p, t = pred == c, target == c
        tp = float(np.logical_and(p, t).sum())
        fp = float(np.logical_and(p, ~t).sum())
        fn = float(np.logical_and(~p, t).sum())
        if tp + fp + fn == 0:
            dice.append(None); iou.append(None); precision.append(None); recall.append(None)
            continue
        dice.append(2 * tp / (2 * tp + fp + fn))
        iou.append(tp / (tp + fp + fn))
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def boundaries_from_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Row index of each layer boundary per column: shape (num_classes - 1, W).

    Boundary k is the first row whose label is >= k+1 (labels increase downwards).
    NaN where the boundary is not present in that column.
    """
    h, w = mask.shape
    out = np.full((num_classes - 1, w), np.nan, dtype=np.float32)
    for k in range(1, num_classes):
        hits = mask >= k
        has = hits.any(axis=0)
        first = np.argmax(hits, axis=0).astype(np.float32)
        out[k - 1, has] = first[has]
    return out


def boundary_mae(pred: np.ndarray, target: np.ndarray, num_classes: int) -> List[Optional[float]]:
    """Mean absolute boundary position error in pixels, per boundary."""
    pb, tb = boundaries_from_mask(pred, num_classes), boundaries_from_mask(target, num_classes)
    errors: List[Optional[float]] = []
    for k in range(num_classes - 1):
        ok = ~np.isnan(pb[k]) & ~np.isnan(tb[k])
        errors.append(float(np.abs(pb[k, ok] - tb[k, ok]).mean()) if ok.any() else None)
    return errors
