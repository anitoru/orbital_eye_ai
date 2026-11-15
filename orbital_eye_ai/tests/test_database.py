import sqlite3

from orbital import db, demo, monitoring


def test_init_db_upgrades_legacy_schema(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE scans (id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_name TEXT NOT NULL,
                    created_at TEXT NOT NULL, width INTEGER, height INTEGER, mean_intensity REAL, status TEXT NOT NULL)""")
    conn.execute("INSERT INTO scans VALUES ('old1', 'a.png', 'old1_a.png', '2026-01-01T00:00:00', 10, 10, 5.0, 'x')")
    conn.commit(); conn.close()
    db.init_db(path)
    conn = db.connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(scans)")}
    assert {"astronaut_id", "thickness_mean_px", "quality_status"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 1  # existing data kept
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"astronauts", "alerts"} <= tables


def test_seed_demo_creates_profiles_and_alerts(app):
    cfg = app.extensions["orbital_cfg"]
    conn = db.connect(cfg.DB_PATH)
    created = demo.seed(conn, cfg)
    assert created == ["AST-001", "AST-002", "AST-003", "AST-004"]
    assert demo.seed(conn, cfg) == []  # idempotent
    alerts = {r["astronaut_id"] for r in conn.execute("SELECT astronaut_id FROM alerts")}
    assert "AST-001" in alerts and "AST-003" in alerts   # simulated thickening
    assert "AST-002" not in alerts and "AST-004" not in alerts  # stable / baseline only
    demo.remove(conn, cfg)
    assert conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 0
    conn.close()


def test_no_alert_without_baseline(app):
    cfg = app.extensions["orbital_cfg"]
    conn = db.connect(cfg.DB_PATH)
    db.insert(conn, "astronauts", {"id": "AST-900", "display_name": "T", "mission": "M",
                                   "mission_status": "in-flight", "created_at": "2026-01-01"})
    db.insert(conn, "scans", {"id": "s1", "filename": "f", "stored_name": "f", "created_at": "2026-01-01",
                              "status": "x", "astronaut_id": "AST-900", "exam_date": "2026-02-01",
                              "exam_phase": "in-flight", "eye": "OD", "quality_status": "good",
                              "seg_method": "heuristic", "seg_valid_fraction": 1.0, "thickness_mean_px": 150})
    assert monitoring.recompute_alerts(conn, "AST-900", 5) == 0
    conn.close()


def test_astronaut_form_validation(client):
    r = client.post("/astronauts/new", data={"id": "bad id!", "display_name": "", "mission": "", "mission_status": "x"})
    assert r.status_code == 200 and b"ID must be" in r.data
    r = client.post("/astronauts/new", data={"id": "ast-010", "display_name": "Crew J", "mission": "SIM",
                                             "mission_status": "pre-flight"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/astronauts/AST-010")
    assert client.get("/astronauts/AST-010").status_code == 200
