import numpy as np
import pytest
from PIL import Image

from orbital import boundaries, demo, imaging
from orbital.ml import metrics


def test_16bit_tiff_not_clipped(tmp_path):
    arr = (np.linspace(0, 65535, 256 * 256).reshape(256, 256)).astype(np.uint16)
    path = tmp_path / "scan.tif"
    Image.fromarray(arr).save(path)
    gray, report = imaging.load_image(path, "scan.tif", 40_000_000, 64)
    assert report.bit_depth == 16
    assert gray.max() > 0.99 and gray.min() < 0.01  # full range preserved, not clipped to 255/65535


def test_rgb_converted_to_gray(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (300, 300), (120, 60, 30)).save(path)
    gray, report = imaging.load_image(path, "scan.png", 40_000_000, 64)
    assert gray.ndim == 2 and report.image_mode == "RGB"
    assert any("Colour" in n for n in report.notes)


def test_blank_image_flagged_poor(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("L", (400, 400), 0).save(path)
    _, report = imaging.load_image(path, "blank.png", 40_000_000, 64)
    assert report.status == "poor"


def test_decompression_bomb_rejected(tmp_path):
    path = tmp_path / "big.png"
    Image.new("L", (3000, 3000), 10).save(path)
    with pytest.raises(imaging.ImageValidationError):
        imaging.load_image(path, "big.png", 1_000_000, 64)


def test_heuristic_tracks_synthetic_thickness_change():
    values = []
    for scale in (1.0, 1.1):
        img = imaging.preprocess(demo.synthetic_bscan(scale, seed=3).astype(np.float32) / 255)
        seg = boundaries.detect_boundaries_heuristic(img)
        assert seg.valid_fraction > 0.9
        values.append(seg.summary["mean"])
    change = (values[1] - values[0]) / values[0] * 100
    assert 7 < change < 13  # synthetic retina was made 10% thicker


def test_heuristic_reports_nothing_on_blank():
    seg = boundaries.detect_boundaries_heuristic(np.zeros((300, 300), dtype=np.float32))
    assert seg.summary["mean"] is None and seg.valid_fraction == 0


def test_metrics_perfect_and_boundaries():
    mask = np.zeros((10, 4), dtype=np.uint8)
    mask[3:6] = 1
    mask[6:] = 2
    scores = metrics.per_class_scores(mask, mask, 3)
    assert scores["dice"] == [1.0, 1.0, 1.0]
    b = metrics.boundaries_from_mask(mask, 3)
    assert list(b[0]) == [3, 3, 3, 3] and list(b[1]) == [6, 6, 6, 6]
    shifted = np.zeros_like(mask)
    shifted[4:7] = 1
    shifted[7:] = 2  # every boundary one row lower
    assert metrics.boundary_mae(shifted, mask, 3) == [1.0, 1.0]
