"""End-to-end processing of one OCT image: validate -> quality -> preprocess -> segment -> measure.

Shared by the upload route and the demo-data seeder so both produce identical records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from . import boundaries, imaging
from .ml import inference


def process_image(path: Path, filename: str, scan_id: str, cfg) -> Dict[str, Any]:
    """Returns column values for the `scans` table. Raises imaging.ImageValidationError."""
    gray, quality = imaging.load_image(path, filename, cfg.MAX_IMAGE_PIXELS, cfg.MIN_IMAGE_SIDE)
    processed = imaging.preprocess(gray)

    derived = Path(cfg.UPLOAD_DIR) / cfg.DERIVED_DIR_NAME
    preview_name = f"{scan_id}_preview.png"
    processed_name = f"{scan_id}_processed.png"
    overlay_name = f"{scan_id}_overlay.png"
    imaging.save_gray_png(gray, derived / preview_name)          # browser-viewable copy (e.g. of TIFF)
    imaging.save_gray_png(processed, derived / processed_name)

    segmenter = inference.load_segmenter(cfg.MODEL_PATH)
    seg = boundaries.segment(processed, segmenter)
    boundaries.render_overlay(processed, seg, derived / overlay_name)
    summary = seg.summary

    if summary["mean"] is None:
        status = "Analysed - boundaries not found"
    elif seg.method == "unet":
        status = "Analysed - U-Net"
    else:
        status = "Analysed - heuristic (non-AI)"

    return {
        "width": quality.width,
        "height": quality.height,
        "mean_intensity": quality.intensity_mean,
        "status": status,
        "image_format": quality.image_format,
        "file_size": quality.file_size,
        "image_mode": quality.image_mode,
        "bit_depth": quality.bit_depth,
        "n_frames": quality.n_frames,
        "intensity_min": quality.intensity_min,
        "intensity_max": quality.intensity_max,
        "intensity_std": quality.intensity_std,
        "quality_status": quality.status,
        "quality_notes": json.dumps(quality.notes + seg.notes),
        "preview_name": preview_name,
        "processed_name": processed_name,
        "overlay_name": overlay_name,
        "seg_method": seg.method,
        "seg_valid_fraction": seg.valid_fraction,
        "thickness_mean_px": summary["mean"],
        "thickness_median_px": summary["median"],
        "thickness_std_px": summary["std"],
        "thickness_profile": json.dumps(seg.profile()),
    }
