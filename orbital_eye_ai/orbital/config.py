"""Application configuration, read from environment variables.

Copy `.env.example` to `.env` to override values locally. Never commit `.env`.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # A random key per start is safe for a prototype; set SECRET_KEY to keep sessions across restarts.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    DB_PATH = Path(os.environ.get("ORBITAL_DB_PATH", BASE_DIR / "orbital_eye.db"))
    UPLOAD_DIR = Path(os.environ.get("ORBITAL_UPLOAD_DIR", BASE_DIR / "uploads"))
    DERIVED_DIR_NAME = "derived"  # previews/overlays live in uploads/derived/

    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(40_000_000)))
    MIN_IMAGE_SIDE = int(os.environ.get("MIN_IMAGE_SIDE", "64"))

    # Path to a trained U-Net checkpoint (.pt). Missing file -> model reported as unavailable.
    MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "retinal_unet.pt"))
    # Optional private download link for the checkpoint (e.g. a Google Drive share link), used by the
    # hosted demo so the weights never have to be committed to the public repository. Keep it secret.
    MODEL_URL = os.environ.get("MODEL_URL", "")

    # Public demo mode: visitors can browse and try uploads, but nothing they do is saved or shared,
    # and deleting / creating profiles is disabled. Demo data is seeded at start-up.
    PUBLIC_DEMO = _bool("PUBLIC_DEMO", False)
    DEMO_UPLOAD_TTL_MIN = int(os.environ.get("DEMO_UPLOAD_TTL_MIN", "30"))

    # DEMONSTRATION threshold: % change in mean total retinal thickness (pixels) vs. baseline.
    # Illustrative only - not clinically validated.
    ALERT_THRESHOLD_PCT = float(os.environ.get("ALERT_THRESHOLD_PCT", "8"))

    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", "5000"))
    DEBUG = _bool("FLASK_DEBUG", False)
    TESTING = False
