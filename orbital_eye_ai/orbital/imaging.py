"""OCT image loading, validation, preprocessing and technical quality assessment.

Everything in this module is TECHNICAL image metadata. None of these values are
clinical or anatomical measurements.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

# extension -> Pillow format names accepted for it
EXTENSION_FORMATS = {
    "png": {"PNG"},
    "jpg": {"JPEG", "MPO"},
    "jpeg": {"JPEG", "MPO"},
    "tif": {"TIFF"},
    "tiff": {"TIFF"},
    "bmp": {"BMP"},
}
ALLOWED_EXTENSIONS = set(EXTENSION_FORMATS)


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable OCT image."""


@dataclass
class QualityReport:
    width: int
    height: int
    image_format: str
    image_mode: str
    bit_depth: int
    n_frames: int
    file_size: int
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    saturated_fraction: float
    dark_fraction: float
    status: str = "good"
    notes: List[str] = field(default_factory=list)


def extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def allowed_file(filename: str) -> bool:
    return extension_of(filename) in ALLOWED_EXTENSIONS


def _bit_depth(image: Image.Image) -> int:
    if image.mode in {"I;16", "I;16B", "I;16L", "I;16N"}:
        return 16
    if image.mode in {"I", "F"}:
        return 32
    if image.mode == "1":
        return 1
    return 8


def to_float_gray(image: Image.Image) -> np.ndarray:
    """Convert any supported mode to a float32 grayscale array in [0, 1].

    16-bit and 32-bit images are scaled by their full numeric range instead of being
    clipped to 8 bits (the original app's convert("L") clipped 16-bit OCT data).
    """
    if image.mode in {"RGB", "RGBA", "P", "LA", "CMYK", "YCbCr"}:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    elif image.mode == "1":
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    elif image.mode == "L":
        arr = np.asarray(image, dtype=np.float32) / 255.0
    else:  # I;16*, I, F
        arr = np.asarray(image, dtype=np.float32)
        if image.mode.startswith("I;16"):
            arr = arr / 65535.0
        else:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return np.clip(np.nan_to_num(arr), 0.0, 1.0).astype(np.float32)


def load_image(path: Path, filename: str, max_pixels: int, min_side: int) -> Tuple[np.ndarray, QualityReport]:
    """Open, validate and convert an uploaded image. Raises ImageValidationError."""
    ext = extension_of(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Unsupported file extension.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                fmt = image.format or "UNKNOWN"
                if fmt not in EXTENSION_FORMATS[ext]:
                    raise ImageValidationError(
                        f"File content is {fmt}, which does not match the .{ext} extension."
                    )
                n_frames = int(getattr(image, "n_frames", 1) or 1)
                if n_frames > 1:
                    # OCT volume stored as multi-page TIFF: analyse the central B-scan.
                    image.seek(n_frames // 2)
                image.load()
                mode = image.mode
                bits = _bit_depth(image)
                width, height = image.size
                gray = to_float_gray(image)
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("Image has too many pixels.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError("The uploaded file is not a valid or readable image.")
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if min(width, height) < min_side:
        raise ImageValidationError(f"Image is too small ({width}x{height}); minimum side is {min_side} px.")

    report = assess_quality(gray, width, height, fmt, mode, bits, n_frames, path.stat().st_size)
    return gray, report


def assess_quality(
    gray: np.ndarray, width: int, height: int, fmt: str, mode: str, bits: int, n_frames: int, file_size: int
) -> QualityReport:
    """Technical image quality heuristics (not clinical)."""
    scale = 255.0
    report = QualityReport(
        width=width,
        height=height,
        image_format=fmt,
        image_mode=mode,
        bit_depth=bits,
        n_frames=n_frames,
        file_size=file_size,
        intensity_min=round(float(gray.min()) * scale, 2),
        intensity_max=round(float(gray.max()) * scale, 2),
        intensity_mean=round(float(gray.mean()) * scale, 2),
        intensity_std=round(float(gray.std()) * scale, 2),
        saturated_fraction=round(float((gray >= 0.995).mean()), 4),
        dark_fraction=round(float((gray <= 0.02).mean()), 4),
    )
    notes = report.notes
    problems = 0
    if report.intensity_std < 10:
        notes.append("Very low contrast: image may be blank or heavily attenuated.")
        problems += 2
    elif report.intensity_std < 20:
        notes.append("Low contrast.")
        problems += 1
    if report.saturated_fraction > 0.05:
        notes.append(f"{report.saturated_fraction:.1%} of pixels are saturated (clipped highlights).")
        problems += 1
    if report.dark_fraction > 0.85:
        notes.append("Image is mostly black; the retina may be out of view or signal is weak.")
        problems += 2
    if min(width, height) < 200:
        notes.append("Low resolution for layer analysis.")
        problems += 1
    if mode in {"RGB", "RGBA", "P"}:
        notes.append("Colour image converted to grayscale; OCT B-scans are normally single-channel.")
    if n_frames > 1:
        notes.append(f"Multi-frame file ({n_frames} frames); the central frame was analysed.")
    # Simple noise estimate: median absolute difference between neighbouring pixels.
    noise = float(np.median(np.abs(np.diff(gray, axis=1)))) * scale
    if noise > 25:
        notes.append("High speckle/noise level.")
        problems += 1

    report.status = "good" if problems == 0 else ("acceptable" if problems <= 1 else "poor")
    if not notes:
        notes.append("No technical issues detected.")
    return report


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Conservative preprocessing for display and boundary detection.

    1. Percentile contrast stretch (1st-99th) - keeps relative intensities.
    2. 3x3 median filter - reduces speckle without moving layer edges noticeably.
    No CLAHE by default: local histogram equalisation can create false edges in OCT.
    """
    lo, hi = np.percentile(gray, [1, 99])
    stretched = np.clip((gray - lo) / (hi - lo), 0, 1) if hi > lo else gray
    img8 = Image.fromarray((stretched * 255).astype(np.uint8))
    filtered = img8.filter(ImageFilter.MedianFilter(size=3))
    return np.asarray(filtered, dtype=np.float32) / 255.0


def save_gray_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path, "PNG")
