import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from orbital import demo  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    application = create_app({
        "TESTING": True,
        "DB_PATH": tmp_path / "test.db",
        "UPLOAD_DIR": tmp_path / "uploads",
        "MODEL_PATH": tmp_path / "missing_model.pt",
        "SECRET_KEY": "test",
    })
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def png_bytes(arr: np.ndarray, fmt: str = "PNG") -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    buf.seek(0)
    return buf


@pytest.fixture()
def synthetic_png():
    return lambda scale=1.0, seed=0: png_bytes(demo.synthetic_bscan(scale, seed))
