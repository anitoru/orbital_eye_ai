"""End-to-end smoke test of the training pipeline on tiny synthetic data.

Skipped when PyTorch is not installed (the web app does not need it).
Runs on GitHub Actions, which installs requirements-ml.txt.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import dataset  # noqa: E402
from orbital import boundaries, demo  # noqa: E402
from orbital.ml import inference  # noqa: E402


def fake_samples(n, seed=0):
    """Synthetic B-scans with 8 boundaries -> masks, mimicking the Duke DME format."""
    out = []
    for i in range(n):
        img = demo.synthetic_bscan(1.0 + 0.02 * i, seed=seed + i, height=128, width=96).astype(np.float32) / 255
        top = np.full(96, 40.0) + i
        bounds = np.stack([top + 6 * k for k in range(8)])
        out.append((img, dataset.boundaries_to_mask(bounds, 128)))
    return out


def test_boundaries_to_mask_labels():
    mask = dataset.boundaries_to_mask(np.array([[2.0, 2.0], [5.0, 6.0]]), 8)
    assert mask[:, 0].tolist() == [0, 0, 1, 1, 1, 2, 2, 2]
    assert mask[:, 1].tolist() == [0, 0, 1, 1, 1, 1, 2, 2]


def test_train_save_and_use_in_app(tmp_path, monkeypatch):
    train = importlib.import_module("train")
    splits = {1: fake_samples(4, 0), 7: fake_samples(2, 10), 9: fake_samples(2, 20)}
    monkeypatch.setattr(train, "load_split", lambda d, subjects, a=1: splits[subjects[0]])

    small = train.TrainConfig(input_height=64, input_width=64, base=8, depth=2, num_workers=0)
    monkeypatch.setattr(train, "TrainConfig", lambda: small)
    monkeypatch.setattr(sys, "argv", ["train.py", "--data_dir", "x", "--out_dir", str(tmp_path),
                                      "--epochs", "2", "--batch_size", "2"])
    train.main()

    ckpt_path = tmp_path / "retinal_unet.pt"
    assert ckpt_path.exists() and (tmp_path / "loss_curves.png").exists()
    metrics = json.loads((tmp_path / "test_metrics.json").read_text())["test"]
    assert len(metrics["dice"]) == 9 and 0.0 <= metrics["mean_dice_layers"] <= 1.0

    # The app can load the checkpoint and segment an image with it
    inference._cache.update(path=None, segmenter=None, status=None)
    status = inference.model_status(ckpt_path)
    assert status["state"] == "loaded" and status["metrics"]["test B-scans"] == 2
    segmenter = inference.load_segmenter(ckpt_path)
    img = demo.synthetic_bscan(1.0, seed=99, height=200, width=300).astype(np.float32) / 255
    seg = boundaries.segment(img, segmenter)
    assert seg.method == "unet" and seg.thickness_px.shape == seg.columns.shape
