"""Load a trained U-Net checkpoint and run inference inside the Flask app.

PyTorch is optional for the web app: if it is not installed or no checkpoint
exists, the model is reported as unavailable and the app falls back to the
classical heuristic (clearly labelled as non-AI).

Checkpoint format (written by training/train.py):
    {"model_state": state_dict, "config": {...}, "version": str,
     "trained_on": str, "metrics": {...} (test-set metrics), "created": str}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "ORBITAL-UNet (retinal layer segmentation)"
SUPPORTED = "Macular OCT B-scan, single-channel grayscale (PNG / JPEG / TIFF)"

_cache: Dict[str, Any] = {"path": None, "segmenter": None, "status": None}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class UNetSegmenter:
    def __init__(self, checkpoint: Dict[str, Any]):
        import torch
        from .unet import UNet

        cfg = checkpoint["config"]
        self.cfg = cfg
        self.num_classes = int(cfg["num_classes"])
        self.input_hw = (int(cfg["input_height"]), int(cfg["input_width"]))
        self.model = UNet(1, self.num_classes, int(cfg.get("base", 32)), int(cfg.get("depth", 4)), 0.0)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.torch = torch

    def predict_mask(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = img.shape
        ih = self.input_hw[0]
        # Keep the B-scan's proportions (the U-Net is fully convolutional); width must be a multiple of 16.
        iw = max(16, int(round(w * ih / h / 16.0)) * 16)
        resized = np.asarray(
            Image.fromarray((img * 255).astype(np.uint8)).resize((iw, ih), Image.BILINEAR), dtype=np.float32
        ) / 255.0
        x = self.torch.from_numpy(resized)[None, None]
        with self.torch.no_grad():
            logits = self.model(x)
        mask = logits.argmax(1)[0].numpy().astype(np.uint8)
        return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))

    def segment(self, img: np.ndarray):
        from ..boundaries import Segmentation
        from .metrics import boundaries_from_mask

        mask = self.predict_mask(img)
        b = boundaries_from_mask(mask, self.num_classes)
        step = max(1, img.shape[1] // 400)
        cols = np.arange(0, img.shape[1], step)
        ilm = b[0, cols]
        bm = b[-1, cols]  # outer retinal boundary (Bruch's membrane in the Duke DME labelling)
        thickness = bm - ilm
        thickness[~(thickness > 0)] = np.nan
        valid = float(np.mean(~np.isnan(thickness))) if thickness.size else 0.0
        notes = ["U-Net segmentation. Research model - not clinically validated; pixel units."]
        return Segmentation("unet", cols, {"ILM": ilm, "RPE": bm}, round(valid, 3), thickness, notes)


def load_segmenter(model_path: Path) -> Optional[UNetSegmenter]:
    """Return a cached segmenter, or None if PyTorch/checkpoint are unavailable."""
    model_path = Path(model_path)
    if _cache["path"] == str(model_path):
        return _cache["segmenter"]
    _cache.update(path=str(model_path), segmenter=None, status=None)
    if not model_path.exists() or not _torch_available():
        return None
    try:
        import torch

        checkpoint = torch.load(str(model_path), map_location="cpu")
        _cache["segmenter"] = UNetSegmenter(checkpoint)
        _cache["status"] = checkpoint
        log.info("Loaded U-Net checkpoint %s", model_path)
    except Exception:
        log.exception("Could not load model checkpoint %s", model_path)
    return _cache["segmenter"]


def model_status(model_path: Path) -> Dict[str, Any]:
    """Honest model status for the UI. Metrics are shown only if stored in the checkpoint."""
    model_path = Path(model_path)
    status: Dict[str, Any] = {
        "name": MODEL_NAME,
        "architecture": "U-Net (PyTorch), 4 levels, 32 base filters",
        "supported": SUPPORTED,
        "checkpoint": model_path.name,
        "version": "-",
        "trained_on": "-",
        "metrics": None,
        "fallback": "Classical intensity heuristic (non-AI) for ILM/RPE detection",
    }
    if not _torch_available():
        status.update(state="unavailable", label="NOT CONNECTED",
                      training="PyTorch is not installed in this environment.")
        return status
    if not model_path.exists():
        status.update(state="untrained", label="UNTRAINED",
                      training=f"No checkpoint found at models/{model_path.name}. Train in Google Colab (training/).")
        return status
    segmenter = load_segmenter(model_path)
    if segmenter is None:
        status.update(state="error", label="LOAD ERROR", training="Checkpoint exists but could not be loaded.")
        return status
    ckpt = _cache["status"] or {}
    status.update(
        state="loaded", label="CONNECTED",
        training="Trained checkpoint loaded.",
        version=str(ckpt.get("version", "-")),
        trained_on=str(ckpt.get("trained_on", "-")),
        metrics=ckpt.get("metrics"),
    )
    return status
