# app/models.py
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_DIR = Path("data")
DB_PATH = DB_DIR / "sorter.db"
THUMBS_DIR = Path("captures") / "thumbs"
CAPTURES_DIR = Path("captures")

_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()

def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA foreign_keys=ON;")
    return _conn

def init_db() -> None:
    conn = _connect()
    with conn:
        # captures: one row per saved image
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_created ON captures(created_at);")

        # ocr_results: text extracted from a capture
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                boxes_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(capture_id) REFERENCES captures(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_capture ON ocr_results(capture_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_created ON ocr_results(created_at);")

        # assignments: decision about where to put a given OCR result
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ocr_id INTEGER NOT NULL,
                slot_id INTEGER,
                row INTEGER,
                col INTEGER,
                x_mm REAL,
                y_mm REAL,
                rule_used TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(ocr_id) REFERENCES ocr_results(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assign_ocr ON assignments(ocr_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assign_created ON assignments(created_at);")

        # moves: queued motion plans (G-code list)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id INTEGER NOT NULL,
                gcode_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                FOREIGN KEY(assignment_id) REFERENCES assignments(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_moves_assignment ON moves(assignment_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_moves_created ON moves(created_at);")

    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

def health_check() -> bool:
    try:
        _connect().execute("SELECT 1;")
        return True
    except Exception:
        return False

# -----------------------------
# Captures
# -----------------------------

def record_capture(path: str) -> int:
    ts = datetime.utcnow().isoformat()
    with _lock:
        conn = _connect()
        cur = conn.execute("INSERT INTO captures(path, created_at) VALUES(?,?)", (path, ts))
        conn.commit()
        return int(cur.lastrowid)

def list_captures(page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
    offset = (page - 1) * page_size
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    rows = conn.execute(
        "SELECT id, path, created_at FROM captures ORDER BY id DESC LIMIT ? OFFSET ?",
        (page_size, offset),
    ).fetchall()
    return total, [dict(r) for r in rows]

def get_capture(capture_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT id, path, created_at FROM captures WHERE id = ?", (capture_id,)
    ).fetchone()
    return dict(row) if row else None

def generate_thumbnail(path: str, max_w: int = 320) -> Optional[str]:
    """
    Best-effort thumbnail generator. Fails silently if OpenCV is missing.
    Returns thumb path or None.
    """
    try:
        import cv2
        import numpy as np
        src = Path(path)
        if not src.exists():
            return None
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, max_w / float(w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        out = THUMBS_DIR / (src.stem + "_thumb.jpg")
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(str(out))
        return str(out)
    except Exception:
        return None

# -----------------------------
# OCR
# -----------------------------

def record_ocr(capture_id: int, result: Dict[str, Any]) -> int:
    text = str(result.get("text", ""))
    confidence = int(result.get("confidence", 0))
    boxes_json = json.dumps(result.get("boxes", {}), ensure_ascii=False)
    ts = datetime.utcnow().isoformat()
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO ocr_results(capture_id, text, confidence, boxes_json, created_at) "
            "VALUES(?,?,?,?,?)",
            (capture_id, text, confidence, boxes_json, ts),
        )
        conn.commit()
        return int(cur.lastrowid)

def get_ocr(ocr_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT id, capture_id, text, confidence, boxes_json, created_at "
        "FROM ocr_results WHERE id = ?",
        (ocr_id,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["boxes"] = json.loads(d.pop("boxes_json") or "{}")
    except Exception:
        d["boxes"] = {}
    return d

# -----------------------------
# Assignment
# -----------------------------

def record_assignment(ocr_id: int, decision: Dict[str, Any]) -> int:
    """
    decision can include: slot_id, row, col, x_mm, y_mm, rule_used
    """
    slot_id = decision.get("slot_id")
    row = decision.get("row")
    col = decision.get("col")
    x_mm = decision.get("x_mm")
    y_mm = decision.get("y_mm")
    rule_used = decision.get("rule_used")
    ts = datetime.utcnow().isoformat()
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO assignments(ocr_id, slot_id, row, col, x_mm, y_mm, rule_used, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                ocr_id,
                int(slot_id) if slot_id is not None else None,
                int(row) if row is not None else None,
                int(col) if col is not None else None,
                float(x_mm) if x_mm is not None else None,
                float(y_mm) if y_mm is not None else None,
                str(rule_used) if rule_used is not None else None,
                ts,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

def get_assignment(assignment_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT id, ocr_id, slot_id, row, col, x_mm, y_mm, rule_used, created_at "
        "FROM assignments WHERE id = ?",
        (assignment_id,),
    ).fetchone()
    return dict(row) if row else None

# -----------------------------
# Moves (planning/execution queue)
# -----------------------------

def enqueue_move(assignment_id: int, gcode: List[str]) -> int:
    ts = datetime.utcnow().isoformat()
    gcode_json = json.dumps(gcode, ensure_ascii=False)
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO moves(assignment_id, gcode_json, status, created_at) "
            "VALUES(?,?,?,?)",
            (assignment_id, gcode_json, "queued", ts),
        )
        conn.commit()
        return int(cur.lastrowid)

def get_move(move_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT id, assignment_id, gcode_json, status, created_at "
        "FROM moves WHERE id = ?",
        (move_id,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["gcode"] = json.loads(d.pop("gcode_json") or "[]")
    except Exception:
        d["gcode"] = []
    return d
