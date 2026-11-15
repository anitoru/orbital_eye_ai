"""Longitudinal comparison and medical-review alerts.

Alerts are generated ONLY when:
  1. the examination has a usable measurement (segmentation found boundaries in
     enough columns and image quality is not "poor"),
  2. a baseline examination of the same astronaut and the same eye exists, and
  3. the change exceeds a configurable DEMONSTRATION threshold.

The threshold is illustrative and NOT clinically validated. An alert means
"recommend review by an ophthalmologist" - never a diagnosis of SANS.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import db

MIN_VALID_FRACTION = 0.6
MEASUREMENT = "Mean ILM-RPE thickness"
UNIT = "px"
RECOMMENDATION = (
    "Recommend review of this examination by an ophthalmologist and comparison with "
    "clinical findings. Demonstration threshold - not a diagnosis."
)


def is_usable(scan: sqlite3.Row) -> bool:
    return (
        scan["thickness_mean_px"] is not None
        and (scan["seg_valid_fraction"] or 0) >= MIN_VALID_FRACTION
        and scan["quality_status"] != "poor"
    )


def baseline_for(
    conn: sqlite3.Connection, astronaut_id: str, eye: Optional[str], method: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """Earliest usable baseline of the same eye, measured with the same method
    (heuristic and U-Net values are not comparable with each other)."""
    rows = db.query(
        conn,
        """SELECT * FROM scans WHERE astronaut_id = ? AND exam_phase = 'baseline'
           AND (eye IS ? OR eye = ?) AND (? IS NULL OR seg_method = ?)
           ORDER BY exam_date ASC, created_at ASC""",
        (astronaut_id, eye, eye, method, method),
    )
    for row in rows:
        if is_usable(row):
            return row
    return None


def pct_change(previous: float, current: float) -> Optional[float]:
    if previous == 0:
        return None
    return round((current - previous) / previous * 100.0, 2)


def compare(baseline: sqlite3.Row, scan: sqlite3.Row) -> Dict[str, Any]:
    prev, cur = baseline["thickness_mean_px"], scan["thickness_mean_px"]
    return {
        "measurement": MEASUREMENT,
        "unit": UNIT,
        "previous": prev,
        "current": cur,
        "difference": round(cur - prev, 2),
        "pct_change": pct_change(prev, cur),
    }


def recompute_alerts(conn: sqlite3.Connection, astronaut_id: str, threshold_pct: float) -> int:
    """Rebuild all alerts for one astronaut. Returns the number of alerts created."""
    conn.execute("DELETE FROM alerts WHERE astronaut_id = ?", (astronaut_id,))
    scans = db.query(
        conn,
        "SELECT * FROM scans WHERE astronaut_id = ? AND exam_phase != 'baseline' ORDER BY exam_date",
        (astronaut_id,),
    )
    created = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for scan in scans:
        if not is_usable(scan):
            continue
        base = baseline_for(conn, astronaut_id, scan["eye"], scan["seg_method"])
        if base is None or base["id"] == scan["id"]:
            continue
        result = compare(base, scan)
        change = result["pct_change"]
        if change is None or abs(change) < threshold_pct:
            continue
        direction = "increase" if change > 0 else "decrease"
        db.insert(conn, "alerts", {
            "astronaut_id": astronaut_id,
            "scan_id": scan["id"],
            "baseline_scan_id": base["id"],
            "category": f"Retinal thickness {direction} vs. baseline",
            "measurement": MEASUREMENT,
            "unit": UNIT,
            "previous_value": result["previous"],
            "current_value": result["current"],
            "pct_change": change,
            "threshold_pct": threshold_pct,
            "exam_date": scan["exam_date"] or scan["created_at"][:10],
            "recommendation": RECOMMENDATION,
            "created_at": now,
        })
        created += 1
    return created


def timeline(conn: sqlite3.Connection, astronaut_id: str) -> List[Dict[str, Any]]:
    """Chronological measurements for charts, with % change vs. the baseline of the same eye."""
    rows = db.query(
        conn,
        "SELECT * FROM scans WHERE astronaut_id = ? ORDER BY exam_date ASC, created_at ASC",
        (astronaut_id,),
    )
    points = []
    for row in rows:
        base = baseline_for(conn, astronaut_id, row["eye"], row["seg_method"])
        usable = is_usable(row)
        change = None
        if usable and base is not None:
            change = pct_change(base["thickness_mean_px"], row["thickness_mean_px"])
        points.append({
            "id": row["id"],
            "date": row["exam_date"] or row["created_at"][:10],
            "phase": row["exam_phase"] or "unspecified",
            "eye": row["eye"] or "-",
            "thickness": row["thickness_mean_px"] if usable else None,
            "pct_change": change,
            "quality": row["quality_status"],
            "usable": usable,
        })
    return points
