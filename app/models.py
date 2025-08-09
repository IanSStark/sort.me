from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DB_PATH = Path("card_sorter.db").resolve()
_CAPTURES_DIR = Path("captures")
_THUMBS_DIR = Path("captures/_thumbs")

# ---------------------------
# Connection helpers
# ---------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def _db() -> sqlite3.Connection:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ---------------------------
# Schema
# ---------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS captures (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    path    TEXT NOT NULL,
    ts      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id  INTEGER NOT NULL,
    text        TEXT NOT NULL,
    confidence  INTEGER NOT NULL,
    boxes_json  TEXT NOT NULL,
    ts          TEXT NOT NULL,
    FOREIGN KEY(capture_id) REFERENCES captures(id)
);

CREATE TABLE IF NOT EXISTS assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ocr_id      INTEGER NOT NULL,
    slot_id     INTEGER,
    rule_used   TEXT NOT NULL,
    overridden  INTEGER NOT NULL DEFAULT 0,
    ts          TEXT NOT NULL,
    FOREIGN KEY(ocr_id) REFERENCES ocr_results(id)
);

-- Define your physical slots here (either explicit XY or row/col for grid mapping)
CREATE TABLE IF NOT EXISTS slots (
    id          INTEGER PRIMARY KEY,      -- slot_id (e.g., 1..N)
    name        TEXT,
    row         INTEGER,
    col         INTEGER,
    x_mm        REAL,
    y_mm        REAL,
    z_pick_mm   REAL,
    z_travel_mm REAL
);

CREATE TABLE IF NOT EXISTS moves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id   INTEGER NOT NULL,
    gcode_json      TEXT NOT NULL,
    executed        INTEGER NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL,
    FOREIGN KEY(assignment_id) REFERENCES assignments(id)
);
"""

# ---------------------------
# Public API expected by main.py
# ---------------------------

def init_db() -> None:
    """Create DB and folders if missing."""
    _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    _THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript(_SCHEMA_SQL)

def health_check() -> bool:
    try:
        with _db() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False

# ---- Captures ----

def record_capture(path: str) -> int:
    ts = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO captures(path, ts) VALUES(?, ?)",
            (str(Path(path).resolve()), ts),
        )
        return int(cur.lastrowid)

def list_captures(page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
    off = (page - 1) * page_size
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
        rows = conn.execute(
            "SELECT id, path, ts FROM captures ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, off),
        ).fetchall()
    return total, [dict(r) for r in rows]

def get_capture(capture_id: int) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            "SELECT id, path, ts FROM captures WHERE id = ?",
            (capture_id,),
        ).fetchone()
        return dict(row) if row else None

def get_capture_path(capture_id: int) -> str:
    row = get_capture(capture_id)
    if not row:
        raise RuntimeError(f"Capture {capture_id} not found")
    return row["path"]

def generate_thumbnail(path: str, max_w: int = 480) -> None:
    """
    Optional: create thumbnails for UI lists. This no-ops if Pillow is not installed.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return  # silently skip if Pillow not available
    try:
        p = Path(path)
        if not p.exists():
            return
        im = Image.open(p)
        w, h = im.size
        if w <= max_w:
            return
        ratio = max_w / float(w)
        im = im.resize((max_w, int(h * ratio)))
        out = _THUMBS_DIR / p.name
        im.save(out, format="JPEG", quality=85)
    except Exception:
        # Thumbnail is optional; ignore errors
        pass

# ---- OCR ----

def record_ocr(capture_id: int, result: Dict[str, Any]) -> int:
    ts = _now()
    text = result.get("text", "") or ""
    conf = int(result.get("confidence", 0))
    boxes_json = json.dumps(result.get("boxes", {}))
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO ocr_results(capture_id, text, confidence, boxes_json, ts) VALUES(?, ?, ?, ?, ?)",
            (capture_id, text, conf, boxes_json, ts),
        )
        return int(cur.lastrowid)

def get_ocr(ocr_id: int) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            "SELECT id, capture_id, text, confidence, boxes_json, ts FROM ocr_results WHERE id = ?",
            (ocr_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        # Keep boxes_json as string; services can json.loads if needed
        return d

def get_ocr_text(ocr_id: int) -> str:
    row = get_ocr(ocr_id)
    if not row:
        raise RuntimeError(f"OCR result {ocr_id} not found")
    return row["text"] or ""

# ---- Assignments ----

def record_assignment(ocr_id: int, decision: Dict[str, Any]) -> int:
    ts = _now()
    slot_id = decision.get("slot_id")
    rule_used = decision.get("rule_used") or "unknown"
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO assignments(ocr_id, slot_id, rule_used, overridden, ts) VALUES(?, ?, ?, ?, ?)",
            (ocr_id, slot_id, rule_used, 0, ts),
        )
        return int(cur.lastrowid)

# ---- Slots / Motion ----

def get_slot_for_assignment(assignment_id: int) -> Optional[Dict[str, Any]]:
    """
    Resolve slot details for the given assignment.
    Returns either explicit XY or row/col mapping for the motion planner.
    """
    with _db() as conn:
        a = conn.execute(
            "SELECT slot_id FROM assignments WHERE id = ?",
            (assignment_id,),
        ).fetchone()
        if not a:
            return None
        slot_id = a["slot_id"]
        if slot_id is None:
            return None
        s = conn.execute(
            "SELECT id, name, row, col, x_mm, y_mm, z_pick_mm, z_travel_mm FROM slots WHERE id = ?",
            (slot_id,),
        ).fetchone()
        if not s:
            # No slot geometry yet → let motion.py raise a clear error
            return {"slot_id": slot_id}
        sd = dict(s)
        # Prepare keys motion.plan_for expects
        out: Dict[str, Any] = {}
        if sd.get("x_mm") is not None and sd.get("y_mm") is not None:
            out["x_mm"] = float(sd["x_mm"])
            out["y_mm"] = float(sd["y_mm"])
        if sd.get("row") is not None and sd.get("col") is not None:
            out["row"] = int(sd["row"])
            out["col"] = int(sd["col"])
        if sd.get("z_pick_mm") is not None:
            out["z_pick_mm"] = float(sd["z_pick_mm"])
        if sd.get("z_travel_mm") is not None:
            out["z_travel_mm"] = float(sd["z_travel_mm"])
        return out

def upsert_slot(
    slot_id: int,
    *,
    name: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    x_mm: Optional[float] = None,
    y_mm: Optional[float] = None,
    z_pick_mm: Optional[float] = None,
    z_travel_mm: Optional[float] = None,
) -> None:
    """
    Utility to seed or update slot geometry.
    Provide either (row,col) or (x_mm,y_mm) or both.
    """
    with _db() as conn:
        cur = conn.execute("SELECT id FROM slots WHERE id = ?", (slot_id,))
        exists = cur.fetchone() is not None
        if exists:
            conn.execute(
                """
                UPDATE slots SET name=?, row=?, col=?, x_mm=?, y_mm=?, z_pick_mm=?, z_travel_mm=?
                WHERE id=?
                """,
                (name, row, col, x_mm, y_mm, z_pick_mm, z_travel_mm, slot_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO slots(id, name, row, col, x_mm, y_mm, z_pick_mm, z_travel_mm)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (slot_id, name, row, col, x_mm, y_mm, z_pick_mm, z_travel_mm),
            )

# ---- Moves ----

def enqueue_move(assignment_id: int, gcode: List[str]) -> int:
    ts = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO moves(assignment_id, gcode_json, executed, ts) VALUES(?, ?, ?, ?)",
            (assignment_id, json.dumps(gcode), 0, ts),
        )
        return int(cur.lastrowid)

def get_move(move_id: int) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            "SELECT id, assignment_id, gcode_json, executed, ts FROM moves WHERE id = ?",
            (move_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["gcode"] = json.loads(d.pop("gcode_json", "[]"))
        except Exception:
            d["gcode"] = []
        d["executed"] = bool(d["executed"])
        return d

def mark_executed(move_id: int) -> None:
    with _db() as conn:
        conn.execute("UPDATE moves SET executed = 1 WHERE id = ?", (move_id,))

# ---------------------------
# Utilities
# ---------------------------

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
