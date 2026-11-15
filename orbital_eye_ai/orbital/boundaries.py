"""Retinal boundary detection and pixel-based thickness measurement.

Two methods:
* "unet"      - a trained U-Net checkpoint (see orbital/ml), used when available.
* "heuristic" - a classical, NON-AI intensity method used as a fallback so the
                monitoring workflow can be demonstrated before a model is trained.

The heuristic finds, per image column:
  ILM  - first strong rise in reflectivity from the vitreous (top of the retina)
  RPE  - the brightest band below the ILM (hyper-reflective RPE/Bruch's complex)
and reports the ILM->RPE distance in PIXELS. It is not validated, fails on
pathology/artefacts, and must not be interpreted as a clinical measurement.
Pixels are only converted to micrometres when the device's axial pixel spacing is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

BOUNDARY_COLORS = {"ILM": (34, 211, 238), "RPE": (251, 191, 36)}


@dataclass
class Segmentation:
    method: str                               # heuristic | unet
    columns: np.ndarray                       # x positions (px) that were analysed
    boundaries: Dict[str, np.ndarray]         # name -> y position per column (NaN = not found)
    valid_fraction: float
    thickness_px: np.ndarray                  # per column, NaN where invalid
    notes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, Optional[float]]:
        valid = self.thickness_px[~np.isnan(self.thickness_px)]
        if valid.size == 0:
            return {"mean": None, "median": None, "std": None}
        return {
            "mean": round(float(valid.mean()), 2),
            "median": round(float(np.median(valid)), 2),
            "std": round(float(valid.std()), 2),
        }

    def profile(self, points: int = 64) -> List[Optional[float]]:
        """Downsampled thickness profile for charts (None where invalid)."""
        if self.thickness_px.size == 0:
            return []
        idx = np.linspace(0, self.thickness_px.size - 1, min(points, self.thickness_px.size)).astype(int)
        return [None if np.isnan(v) else round(float(v), 1) for v in self.thickness_px[idx]]


def _box_smooth(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius < 1:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")
    csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    csum = np.insert(csum, 0, 0, axis=axis)
    n = arr.shape[axis]
    hi = np.take(csum, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * radius + 1)).astype(np.float32)


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    out = values.copy()
    half = window // 2
    for i in range(values.size):
        seg = values[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size and not np.isnan(values[i]):
            out[i] = np.median(seg)
    return out


def detect_boundaries_heuristic(img: np.ndarray, max_columns: int = 400) -> Segmentation:
    """img: preprocessed float image in [0, 1], shape (H, W)."""
    h, w = img.shape
    step = max(1, w // max_columns)
    cols = np.arange(0, w, step)
    sub = img[:, cols]
    # smooth vertically (twice = approx. Gaussian) and slightly across columns
    radius = max(1, h // 150)
    smooth = _box_smooth(_box_smooth(sub, radius, axis=0), radius, axis=0)
    smooth = _box_smooth(smooth, 1, axis=1)

    background = np.percentile(smooth, 20, axis=0)
    peak = np.percentile(smooth, 99.5, axis=0)
    contrast = peak - background
    threshold = background + 0.45 * contrast
    min_gap = max(3, int(h * 0.02))
    max_span = int(h * 0.6)

    ilm = np.full(cols.size, np.nan, dtype=np.float32)
    rpe = np.full(cols.size, np.nan, dtype=np.float32)
    for j in range(cols.size):
        if contrast[j] < 0.15:
            continue
        above = np.nonzero(smooth[:, j] > threshold[j])[0]
        if above.size == 0:
            continue
        top = int(above[0])
        lo, hi = top + min_gap, min(h, top + max_span)
        if hi - lo < 2:
            continue
        bottom = lo + int(np.argmax(smooth[lo:hi, j]))
        if smooth[bottom, j] < threshold[j]:
            continue
        ilm[j], rpe[j] = top, bottom

    window = 9 if cols.size >= 30 else 3
    ilm = _median_filter_1d(ilm, window)
    rpe = _median_filter_1d(rpe, window)
    thickness = rpe - ilm
    thickness[thickness <= 0] = np.nan
    valid_fraction = float(np.mean(~np.isnan(thickness))) if thickness.size else 0.0

    notes = ["Classical intensity heuristic (non-AI). Not validated; pixel units only."]
    if valid_fraction < 0.6:
        notes.append(f"Boundaries found in only {valid_fraction:.0%} of columns - result unreliable.")
    return Segmentation("heuristic", cols, {"ILM": ilm, "RPE": rpe}, round(valid_fraction, 3), thickness, notes)


def segment(img: np.ndarray, model=None) -> Segmentation:
    """Use the U-Net if a loaded model is provided, otherwise the heuristic fallback."""
    if model is not None:
        try:
            return model.segment(img)
        except Exception as exc:  # pragma: no cover - logged and fall back
            seg = detect_boundaries_heuristic(img)
            seg.notes.append(f"U-Net inference failed ({exc}); heuristic used instead.")
            return seg
    return detect_boundaries_heuristic(img)


def render_overlay(img: np.ndarray, seg: Segmentation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(base)
    line_w = max(2, base.height // 200)
    for name, ys in seg.boundaries.items():
        color = BOUNDARY_COLORS.get(name, (236, 72, 153))
        run: List[tuple] = []
        for x, y in zip(seg.columns, ys):
            if np.isnan(y):
                if len(run) > 1:
                    draw.line(run, fill=color, width=line_w)
                run = []
            else:
                run.append((int(x), int(y)))
        if len(run) > 1:
            draw.line(run, fill=color, width=line_w)
    base.save(path, "PNG")
