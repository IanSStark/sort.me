# app/services/motion.py
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import serial  # pyserial

from app import models

logger = logging.getLogger("motion")

# =========================
# Configuration / State
# =========================

@dataclass
class GridConfig:
    origin_mm: Tuple[float, float] = (0.0, 0.0)
    pitch_mm: Tuple[float, float] = (70.0, 95.0)
    rows: int = 5
    cols: int = 10
    z_pick: float = -2.0
    z_travel: float = 15.0
    z_safe: float = 20.0


@dataclass
class MotionState:
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False
    feed_xy: int = 3000
    feed_z: int = 600
    homing_sequence: List[str] = field(default_factory=lambda: ["G28"])
    grid: GridConfig = field(default_factory=GridConfig)

    # runtime
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _ser: Optional[serial.Serial] = None
    _connected: bool = False
    _homed_once: bool = False
    _last_connect_err: Optional[str] = None


_state = MotionState()

# Marlin “ok” handling
_OK_RE = re.compile(rb"(^|\s)ok(\s|$)", re.IGNORECASE)

# =========================
# Public API (used by main)
# =========================

def init(cfg: Optional[Dict] = None) -> None:
    """
    Initialize motion service. Does not open the serial port yet.
    Expected cfg keys:
      port, baud, enabled, homing_sequence, feedrates{xy,z}, grid{...}
    """
    global _state
    if cfg:
        _state.port = cfg.get("port", _state.port)
        _state.baud = int(cfg.get("baud", _state.baud))
        _state.enabled = bool(cfg.get("enabled", _state.enabled))

        # feedrates
        fr = cfg.get("feedrates") or {}
        _state.feed_xy = int(fr.get("xy", _state.feed_xy))
        _state.feed_z = int(fr.get("z", _state.feed_z))

        # homing
        hs = cfg.get("homing_sequence")
        if isinstance(hs, list) and hs:
            _state.homing_sequence = [str(x) for x in hs]

        # grid
        grid = cfg.get("grid") or {}
        gz = grid.get("z") or {}
        _state.grid = GridConfig(
            origin_mm=tuple(grid.get("origin_mm", _state.grid.origin_mm)),
            pitch_mm=tuple(grid.get("pitch_mm", _state.grid.pitch_mm)),
            rows=int(grid.get("rows", _state.grid.rows)),
            cols=int(grid.get("cols", _state.grid.cols)),
            z_pick=float(gz.get("pick", _state.grid.z_pick)),
            z_travel=float(gz.get("travel", _state.grid.z_travel)),
            z_safe=float(gz.get("safe", _state.grid.z_safe)),
        )

    logger.info(
        "Motion init: port=%s baud=%s enabled=%s feed_xy=%s feed_z=%s grid=%s",
        _state.port, _state.baud, _state.enabled, _state.feed_xy, _state.feed_z, _state.grid
    )


def shutdown() -> None:
    """Close serial if open."""
    with _state._lock:
        try:
            if _state._ser and _state._ser.is_open:
                _state._ser.close()
        except Exception:
            logger.warning("Error closing serial", exc_info=True)
        finally:
            _state._ser = None
            _state._connected = False
            _state._homed_once = False
            logger.info("Motion shutdown complete")


def enabled() -> bool:
    return _state.enabled


def set_enabled(value: bool) -> bool:
    with _state._lock:
        _state.enabled = bool(value)
        if not _state.enabled:
            # If disabling, drop connection for safety
            try:
                if _state._ser and _state._ser.is_open:
                    _state._ser.close()
            except Exception:
                logger.warning("Error closing port on disable", exc_info=True)
            finally:
                _state._ser = None
                _state._connected = False
                _state._homed_once = False
        return _state.enabled


def port_or_none() -> Optional[str]:
    return _state.port or None


def plan_for(assignment_id: int) -> List[str]:
    """
    Build a safe, idempotent G-code plan for the given assignment.
    The plan:
      - Absolute, mm
      - Raise to Z_safe
      - Rapid to XY slot center
      - Feed down to Z_pick
      - (comment placeholder for pick/place)
      - Raise to Z_travel
    """
    slot = models.get_slot_for_assignment(assignment_id)
    if not slot:
        raise ValueError(f"No slot data for assignment {assignment_id}")

    pose = _resolve_slot_pose(slot)
    x, y = pose["x_mm"], pose["y_mm"]
    z_pick = pose["z_pick_mm"]
    z_travel = pose["z_travel_mm"]
    z_safe = _state.grid.z_safe

    g = [
        "G90 ; absolute positioning",
        "G21 ; millimeters",
        f"G0 Z{z_safe:.3f}",  # clear before XY moves
        f"G0 X{x:.3f} Y{y:.3f} F{_state.feed_xy}",
        f"G1 Z{z_pick:.3f} F{_state.feed_z}",
        "; --- perform pick/place here (external plunger control) ---",
        f"G0 Z{z_travel:.3f} F{_state.feed_z}",
    ]
    return g


def execute(move_id: int) -> Dict[str, object]:
    """
    Execute a previously queued move (list of G-code lines from DB).
    Returns { "ok": True } on success, or { "ok": False, "error": "..."}.
    """
    if not _state.enabled:
        return {"ok": False, "error": "Motion disabled"}

    move = models.get_move(move_id)
    if not move:
        return {"ok": False, "error": f"Move {move_id} not found"}

    gcode: List[str] = list(move.get("gcode") or [])
    if not gcode:
        return {"ok": False, "error": "Move has no G-code"}

    try:
        with _state._lock:
            _ensure_connected_and_homed()

            # Safety header on every job
            header = [
                "G90",  # absolute
                "G21",  # mm
                "M400", # finish moves
            ]
            _send_program(header + gcode + ["M400"])

        models.mark_executed(move_id)
        return {"ok": True}
    except Exception as e:
        logger.exception("Execute failed for move %s", move_id)
        return {"ok": False, "error": str(e)}


# =========================
# Planning helpers
# =========================

def _resolve_slot_pose(slot: Dict) -> Dict[str, float]:
    """
    Resolve slot coordinates and Z setpoints.
    Accepts either explicit coordinates or logical row/col for grid mapping.
    """
    # Prefer explicit coordinates if available
    if all(k in slot for k in ("x_mm", "y_mm")):
        x = float(slot["x_mm"])
        y = float(slot["y_mm"])
    else:
        # Compute from row/col and grid
        if not all(k in slot for k in ("row", "col")):
            raise ValueError("Slot must provide either (x_mm,y_mm) or (row,col)")
        row = int(slot["row"])
        col = int(slot["col"])
        if row < 0 or row >= _state.grid.rows or col < 0 or col >= _state.grid.cols:
            raise ValueError(f"Slot row/col out of bounds: ({row},{col})")
        x = _state.grid.origin_mm[0] + col * _state.grid.pitch_mm[0]
        y = _state.grid.origin_mm[1] + row * _state.grid.pitch_mm[1]

    # Z parameters: per-slot override if present, else grid defaults
    z_pick = float(slot.get("z_pick_mm", _state.grid.z_pick))
    z_travel = float(slot.get("z_travel_mm", _state.grid.z_travel))

    return {"x_mm": x, "y_mm": y, "z_pick_mm": z_pick, "z_travel_mm": z_travel}


# =========================
# Serial / Marlin helpers
# =========================

def _ensure_connected_and_homed() -> None:
    """Open serial if needed and run homing once per session."""
    if _state._ser and _state._ser.is_open:
        return

    # (Re)open
    try:
        ser = serial.Serial(_state.port, _state.baud, timeout=2, write_timeout=2)
        _state._ser = ser
        _state._connected = True
        _state._last_connect_err = None
        logger.info("Opened serial port %s @ %s", _state.port, _state.baud)
        _drain_until_ok(ser, max_lines=10, timeout_s=2.5)
    except Exception as e:
        _state._connected = False
        _state._last_connect_err = str(e)
        raise RuntimeError(f"Failed to open {_state.port}: {e}")

    # Home once per connection (unless disabled by empty homing_sequence)
    if _state.homing_sequence and not _state._homed_once:
        _send_program(_state.homing_sequence)
        _state._homed_once = True


def _send_program(lines: List[str]) -> None:
    """
    Send a list of G-code lines, waiting for 'ok' after each.
    Strips comments (;) and empty lines, enforces timeouts.
    """
    if not (_state._ser and _state._ser.is_open):
        raise RuntimeError("Serial not connected")

    ser = _state._ser
    for raw in lines:
        line = _sanitize_gcode_line(raw)
        if not line:
            continue
        _write_and_wait_ok(ser, line)


def _sanitize_gcode_line(line: str) -> Optional[str]:
    # Strip comments starting with ';' and inline spaces
    line = (line or "").strip()
    if not line:
        return None
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    if not line:
        return None
    # Ensure newline at send-time; here we return just the content
    return line


def _write_and_wait_ok(ser: serial.Serial, line: str, timeout_s: float = 5.0) -> None:
    """
    Write a single line and wait for an 'ok' from Marlin.
    Raises RuntimeError on timeout.
    """
    msg = (line + "\n").encode("ascii", errors="ignore")
    ser.write(msg)
    ser.flush()

    start = time.time()
    buf = b""
    while True:
        if (time.time() - start) > timeout_s:
            raise RuntimeError(f"Timeout waiting for ok after: {line}")
        try:
            chunk = ser.readline()  # reads until '\n' or timeout
        except Exception:
            chunk = b""
        if not chunk:
            continue
        buf += chunk
        if _OK_RE.search(chunk):
            # Optional: log intermediate responses for debugging
            if buf.strip():
                _log_device_lines(buf)
            return
        # Collect multi-line responses until an "ok" arrives
        if len(buf) > 4096:  # avoid unbounded buffer
            _log_device_lines(buf)
            buf = b""


def _drain_until_ok(ser: serial.Serial, max_lines: int = 10, timeout_s: float = 2.0) -> None:
    """
    Drain initial boot banner or buffered responses until an ok or limits reached.
    """
    start = time.time()
    lines_seen = 0
    while lines_seen < max_lines and (time.time() - start) < timeout_s:
        try:
            line = ser.readline()
        except Exception:
            break
        if not line:
            continue
        lines_seen += 1
        if _OK_RE.search(line):
            break


def _log_device_lines(blob: bytes) -> None:
    try:
        for ln in blob.splitlines():
            s = ln.decode("utf-8", "ignore").strip()
            if s:
                logger.debug("DEV> %s", s)
    except Exception:
        pass
