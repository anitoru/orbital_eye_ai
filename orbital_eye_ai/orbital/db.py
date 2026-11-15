"""SQLite database access and schema.

The schema is created (and older databases upgraded) every time the app starts,
so `python app.py`, `flask run` and gunicorn all work on a fresh checkout.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS astronauts (
    id TEXT PRIMARY KEY,               -- e.g. AST-001 (fictional / anonymised)
    display_name TEXT NOT NULL,
    mission TEXT NOT NULL,
    mission_status TEXT NOT NULL,      -- pre-flight | in-flight | post-flight | completed
    notes TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    mean_intensity REAL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    astronaut_id TEXT NOT NULL,
    scan_id TEXT NOT NULL,
    baseline_scan_id TEXT NOT NULL,
    category TEXT NOT NULL,
    measurement TEXT NOT NULL,
    unit TEXT NOT NULL,
    previous_value REAL NOT NULL,
    current_value REAL NOT NULL,
    pct_change REAL,
    threshold_pct REAL NOT NULL,
    exam_date TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (astronaut_id) REFERENCES astronauts(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
"""

# Columns added to the original `scans` table (kept for backwards compatibility).
SCAN_EXTRA_COLUMNS = {
    "astronaut_id": "TEXT",
    "exam_date": "TEXT",
    "exam_phase": "TEXT",          # baseline | in-flight | post-flight
    "eye": "TEXT",                 # OD | OS
    "image_format": "TEXT",
    "file_size": "INTEGER",
    "image_mode": "TEXT",
    "bit_depth": "INTEGER",
    "n_frames": "INTEGER",
    "intensity_min": "REAL",
    "intensity_max": "REAL",
    "intensity_std": "REAL",
    "quality_status": "TEXT",      # good | acceptable | poor
    "quality_notes": "TEXT",       # JSON list of strings
    "preview_name": "TEXT",
    "processed_name": "TEXT",
    "overlay_name": "TEXT",
    "seg_method": "TEXT",          # heuristic | unet | none
    "seg_valid_fraction": "REAL",
    "thickness_mean_px": "REAL",
    "thickness_median_px": "REAL",
    "thickness_std_px": "REAL",
    "thickness_profile": "TEXT",   # JSON list (per-column, downsampled)
    "is_demo": "INTEGER DEFAULT 0",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_scans_astronaut ON scans(astronaut_id, exam_date)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_astronaut ON alerts(astronaut_id)",
]


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
        for column, sql_type in SCAN_EXTRA_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE scans ADD COLUMN {column} {sql_type}")
        for statement in INDEXES:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def query(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()


def query_one(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchone()


def insert(conn: sqlite3.Connection, table: str, values: Dict[str, Any]) -> None:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))
