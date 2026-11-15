from orbital import db, demo


def test_full_workflow_pages_with_demo_data(app, client):
    cfg = app.extensions["orbital_cfg"]
    conn = db.connect(cfg.DB_PATH)
    demo.seed(conn, cfg)
    ids = [r["id"] for r in conn.execute("SELECT id FROM scans WHERE astronaut_id='AST-001' ORDER BY exam_date")]
    conn.close()

    assert b"AST-001" in client.get("/").data
    detail = client.get("/astronauts/AST-001")
    assert detail.status_code == 200 and b"thicknessChart" in detail.data and b"changeChart" in detail.data
    assert b"Retinal thickness increase" in client.get("/alerts").data
    scan = client.get(f"/scans/{ids[-1]}")
    assert scan.status_code == 200 and b"Change vs. baseline" in scan.data
    cmp = client.get(f"/compare?scan={ids[0]}&scan={ids[-1]}")
    assert cmp.status_code == 200 and b"%" in cmp.data
    assert client.get("/history?astronaut=AST-001&from=2026-03-01").status_code == 200


def test_upload_for_astronaut_creates_alert(app, client, synthetic_png):
    client.post("/astronauts/new", data={"id": "AST-050", "display_name": "Crew", "mission": "SIM",
                                         "mission_status": "in-flight"})
    for scale, phase, date in [(1.0, "baseline", "2026-01-01"), (1.15, "in-flight", "2026-04-01")]:
        r = client.post("/analysis", content_type="multipart/form-data", data={
            "oct_image": (synthetic_png(scale, seed=7), "s.png"), "astronaut_id": "AST-050",
            "exam_date": date, "exam_phase": phase, "eye": "OD"})
        assert r.status_code == 302 and "/scans/" in r.headers["Location"]
    conn = db.connect(app.extensions["orbital_cfg"].DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE astronaut_id='AST-050'").fetchone()[0] == 1
    conn.close()


def test_delete_scan_removes_alert(app, client):
    cfg = app.extensions["orbital_cfg"]
    conn = db.connect(cfg.DB_PATH)
    demo.seed(conn, cfg)
    alert = conn.execute("SELECT scan_id FROM alerts WHERE astronaut_id='AST-001'").fetchone()
    conn.close()
    r = client.post(f"/scans/{alert['scan_id']}/delete")
    assert r.status_code == 302
    conn = db.connect(cfg.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE scan_id=?", (alert["scan_id"],)).fetchone()[0] == 0
    conn.close()
