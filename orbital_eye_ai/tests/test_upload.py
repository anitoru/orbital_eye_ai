import io

from conftest import png_bytes
import numpy as np


def upload(client, file, name, **form):
    data = {"oct_image": (file, name), "exam_date": form.pop("exam_date", "2026-01-01"), **form}
    return client.post("/analysis", data=data, content_type="multipart/form-data")


def test_pages_load_on_fresh_database(client):
    for path in ["/", "/analysis", "/history", "/astronauts", "/astronauts/new", "/alerts", "/model"]:
        assert client.get(path).status_code == 200, path


def test_valid_upload_redirects_to_detail(client, synthetic_png):
    r = upload(client, synthetic_png(), "scan.png")
    assert r.status_code == 302 and "/scans/" in r.headers["Location"]
    page = client.get(r.headers["Location"])
    assert page.status_code == 200
    assert b"Quality report" in page.data and b"px" in page.data


def test_rejects_wrong_extension(client):
    r = upload(client, io.BytesIO(b"x"), "evil.exe")
    assert r.status_code == 302 and r.headers["Location"].endswith("/analysis")


def test_rejects_fake_image(client, app):
    r = upload(client, io.BytesIO(b"not an image at all"), "fake.png")
    assert r.status_code == 302
    assert not list((app.config["UPLOAD_DIR"]).glob("*fake*"))


def test_rejects_extension_format_mismatch(client):
    jpeg = png_bytes(np.full((300, 300), 100, dtype=np.uint8), "JPEG")
    r = upload(client, jpeg, "scan.png", )
    assert r.status_code == 302
    follow = client.get("/analysis")
    assert b"does not match" in follow.data


def test_rejects_invalid_date(client, synthetic_png):
    r = upload(client, synthetic_png(), "scan.png", exam_date="not-a-date")
    assert r.status_code == 302 and r.headers["Location"].endswith("/analysis")


def test_rejects_too_small_image(client):
    r = upload(client, png_bytes(np.zeros((20, 20), dtype=np.uint8)), "tiny.png")
    assert r.status_code == 302 and r.headers["Location"].endswith("/analysis")


def test_too_large_upload(client, app):
    app.config["MAX_CONTENT_LENGTH"] = 1000
    r = upload(client, io.BytesIO(b"0" * 5000), "big.png")
    assert r.status_code == 302


def test_path_traversal_blocked(client):
    assert client.get("/uploads/../app.py").status_code == 404


def test_unknown_scan_404(client):
    assert client.get("/scans/doesnotexist").status_code == 404
