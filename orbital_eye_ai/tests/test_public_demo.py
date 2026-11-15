import io
import os
import time
from pathlib import Path

import pytest

from app import cleanup_demo_files, create_app
from orbital import db, demo
from orbital.ml import download


@pytest.fixture()
def demo_app(tmp_path):
    application = create_app({
        "TESTING": True, "PUBLIC_DEMO": True, "SECRET_KEY": "t",
        "DB_PATH": tmp_path / "demo.db", "UPLOAD_DIR": tmp_path / "uploads",
        "MODEL_PATH": tmp_path / "none.pt",
    })
    cfg = application.extensions["orbital_cfg"]
    conn = db.connect(cfg.DB_PATH)
    demo.seed(conn, cfg)
    conn.close()
    return application


def counts(app):
    conn = db.connect(app.extensions["orbital_cfg"].DB_PATH)
    try:
        return (conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM astronauts").fetchone()[0])
    finally:
        conn.close()


def test_banner_and_hidden_buttons(demo_app):
    c = demo_app.test_client()
    page = c.get("/astronauts").data
    assert b"Public demo" in page and b"New profile" not in page
    conn = db.connect(demo_app.extensions["orbital_cfg"].DB_PATH)
    scan_id = conn.execute("SELECT id FROM scans LIMIT 1").fetchone()[0]
    conn.close()
    assert b"Delete examination" not in c.get(f"/scans/{scan_id}").data


def test_delete_and_new_profile_blocked(demo_app):
    c = demo_app.test_client()
    before = counts(demo_app)
    conn = db.connect(demo_app.extensions["orbital_cfg"].DB_PATH)
    scan_id = conn.execute("SELECT id FROM scans LIMIT 1").fetchone()[0]
    conn.close()
    assert c.post(f"/scans/{scan_id}/delete").status_code == 302
    assert c.get("/astronauts/new").status_code == 302
    r = c.post("/astronauts/new", data={"id": "AST-777", "display_name": "x", "mission": "y",
                                        "mission_status": "pre-flight"})
    assert r.status_code == 302
    assert counts(demo_app) == before


def test_visitor_upload_is_shown_but_not_saved(demo_app, synthetic_png):
    c = demo_app.test_client()
    before = counts(demo_app)
    r = c.post("/analysis", content_type="multipart/form-data", data={
        "oct_image": (synthetic_png(1.2, seed=5), "mine.png"), "astronaut_id": "AST-002",
        "exam_date": "2026-08-01", "exam_phase": "in-flight", "eye": "OD"})
    assert r.status_code == 200
    assert b"shown only to you" in r.data and b"Change vs. baseline" in r.data
    assert counts(demo_app) == before
    up = demo_app.extensions["orbital_cfg"].UPLOAD_DIR
    assert not list(up.glob("tmp-*"))                 # original deleted
    derived = list((up / "derived").glob("tmp-*"))
    assert len(derived) == 3                          # preview, processed, overlay kept for display


def test_cleanup_removes_old_visitor_files(demo_app):
    cfg = demo_app.extensions["orbital_cfg"]
    old = cfg.UPLOAD_DIR / "derived" / "tmp-old_overlay.png"
    old.write_bytes(b"x")
    past = time.time() - (cfg.DEMO_UPLOAD_TTL_MIN + 5) * 60
    os.utime(old, (past, past))
    keep = cfg.UPLOAD_DIR / "derived" / "tmp-new_overlay.png"
    keep.write_bytes(b"x")
    cleanup_demo_files(cfg)
    assert not old.exists() and keep.exists()


def test_google_drive_link_conversion():
    share = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp_qrs-TUV/view?usp=sharing"
    assert download.google_drive_id(share) == "1AbCdEfGhIjKlMnOp_qrs-TUV"
    assert "confirm=t" in download.direct_url(share)
    assert download.direct_url("https://example.org/m.pt") == "https://example.org/m.pt"


def test_ensure_model_rejects_html(tmp_path, monkeypatch):
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"<html>login</html>"))
    target = tmp_path / "m.pt"
    assert download.ensure_model(target, "https://example.org/m.pt") is False and not target.exists()
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"PK\x03\x04data"))
    assert download.ensure_model(target, "https://example.org/m.pt") is True and target.exists()
