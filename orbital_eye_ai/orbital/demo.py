"""Fictional astronauts and SYNTHETIC OCT-like images for demonstrations.

Nothing here is real patient or astronaut data. The images are generated
mathematically (layered bands + speckle noise) and only imitate the general
appearance of a macular OCT B-scan so the full workflow can be demonstrated.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from . import db, monitoring, pipeline

DEMO_ASTRONAUTS = [
    # id, display name, mission, status, notes, exams: (date, phase, eye, thickness scale)
    ("AST-001", "Demo Astronaut A", "ARTEMIS-SIM 1", "in-flight",
     "Fictional profile. Synthetic scans simulate progressive retinal thickening.",
     [("2026-01-12", "baseline", "OD", 1.00), ("2026-03-20", "in-flight", "OD", 1.03),
      ("2026-05-18", "in-flight", "OD", 1.07), ("2026-07-15", "in-flight", "OD", 1.12)]),
    ("AST-002", "Demo Astronaut B", "ARTEMIS-SIM 1", "in-flight",
     "Fictional profile. Synthetic scans simulate a stable retina.",
     [("2026-01-14", "baseline", "OD", 1.00), ("2026-03-22", "in-flight", "OD", 1.01),
      ("2026-05-20", "in-flight", "OD", 0.99), ("2026-07-17", "in-flight", "OD", 1.02)]),
    ("AST-003", "Demo Astronaut C", "LUNAR GATEWAY-SIM", "post-flight",
     "Fictional profile. Thickening in flight, partial recovery after return.",
     [("2025-09-02", "baseline", "OS", 1.00), ("2025-12-10", "in-flight", "OS", 1.05),
      ("2026-03-01", "in-flight", "OS", 1.10), ("2026-06-05", "post-flight", "OS", 1.04)]),
    ("AST-004", "Demo Astronaut D", "MARS TRANSIT-SIM", "pre-flight",
     "Fictional profile. Baseline only - no comparison possible yet.",
     [("2026-10-01", "baseline", "OD", 1.00)]),
]


def synthetic_bscan(scale: float = 1.0, seed: int = 0, height: int = 496, width: int = 768) -> np.ndarray:
    """Generate an OCT-like B-scan (uint8). `scale` multiplies the retinal thickness."""
    rng = np.random.default_rng(seed)
    x = np.arange(width)
    centre = width / 2 + rng.normal(0, 8)
    rpe = 270 - 0.0001 * (x - centre) ** 2 + rng.normal(0, 2)              # gently curved outer retina (RPE)
    base_thick = 105 - 45 * np.exp(-((x - centre) ** 2) / (2 * 55 ** 2))     # foveal pit (thinner centre)
    thick = base_thick * scale
    ilm = rpe - thick                                                       # pit appears as a dip in the ILM

    y = np.arange(height)[:, None]
    img = np.full((height, width), 0.06, dtype=np.float32)
    rel = (y - ilm[None, :]) / np.maximum(thick[None, :], 1)               # 0 at ILM, 1 at RPE
    inside = (rel >= 0) & (rel <= 1)
    # inner retina brighter (NFL), darker nuclear layers, bright photoreceptor/RPE bands
    profile = (0.55 * np.exp(-((rel - 0.05) / 0.06) ** 2) + 0.30 * np.exp(-((rel - 0.35) / 0.12) ** 2)
               + 0.15 + 0.45 * np.exp(-((rel - 0.82) / 0.03) ** 2))
    img = np.where(inside, profile, img)
    rpe_band = np.exp(-((y - rpe[None, :]) / 4.0) ** 2)
    img += 0.85 * rpe_band
    below = y > rpe[None, :] + 4
    choroid = 0.35 * np.exp(-(y - rpe[None, :] - 4) / 60.0)
    img = np.where(below, img + choroid, img)
    speckle = rng.gamma(4.0, 0.25, size=img.shape).astype(np.float32)       # multiplicative speckle
    img = img * speckle + rng.normal(0, 0.02, size=img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def seed(conn, cfg) -> List[str]:
    """Insert demo astronauts and synthetic scans if they are not present yet."""
    created = []
    upload_dir = Path(cfg.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for n, (aid, name, mission, status, notes, exams) in enumerate(DEMO_ASTRONAUTS):
        if db.query_one(conn, "SELECT id FROM astronauts WHERE id = ?", (aid,)):
            continue
        db.insert(conn, "astronauts", {
            "id": aid, "display_name": name, "mission": mission, "mission_status": status,
            "notes": notes, "is_demo": 1, "created_at": now,
        })
        for k, (date, phase, eye, scale) in enumerate(exams):
            scan_id = uuid.uuid4().hex[:12]
            filename = f"SYNTHETIC_{aid}_{date}_{eye}.png"
            stored = f"{scan_id}_{filename}"
            Image.fromarray(synthetic_bscan(scale, seed=n * 100 + k)).save(upload_dir / stored, "PNG")
            values = pipeline.process_image(upload_dir / stored, filename, scan_id, cfg)
            values.update({
                "id": scan_id, "filename": filename, "stored_name": stored, "created_at": now,
                "astronaut_id": aid, "exam_date": date, "exam_phase": phase, "eye": eye, "is_demo": 1,
            })
            db.insert(conn, "scans", values)
        monitoring.recompute_alerts(conn, aid, cfg.ALERT_THRESHOLD_PCT)
        created.append(aid)
    conn.commit()
    return created


def remove(conn, cfg) -> int:
    """Delete all demo astronauts, their scans, alerts and files."""
    rows = db.query(conn, "SELECT stored_name, preview_name, processed_name, overlay_name FROM scans WHERE is_demo = 1")
    derived = Path(cfg.UPLOAD_DIR) / cfg.DERIVED_DIR_NAME
    for r in rows:
        (Path(cfg.UPLOAD_DIR) / r["stored_name"]).unlink(missing_ok=True)
        for name in (r["preview_name"], r["processed_name"], r["overlay_name"]):
            if name:
                (derived / name).unlink(missing_ok=True)
    conn.execute("DELETE FROM alerts WHERE astronaut_id IN (SELECT id FROM astronauts WHERE is_demo = 1)")
    conn.execute("DELETE FROM scans WHERE is_demo = 1")
    conn.execute("DELETE FROM astronauts WHERE is_demo = 1")
    conn.commit()
    return len(rows)
