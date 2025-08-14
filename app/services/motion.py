# app/services/motion.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("motion")

# =========================
# Types / State
# =========================

@dataclass
class GridConfig:
    origin_mm: Tuple[float, float] = (0.0, 0.0)     # (x0, y0)
    pitch_mm: Tuple[float, float] = (70.0, 95.0)    # (dx, dy)
    rows: int = 5
    cols: int = 10
    z_pick: float = -2.0
    z_travel: float = 15.0
    z_safe: float = 20.0

@dataclass
class State:
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False
    homing_sequence: List[str] = None
    feed_xy: int = 3000
    feed_z: int = 600
    grid: GridConfig = GridConfig()

_state = State(homing_sequence=["G28"])

# Serial backend (optional, only if enabled)
_ser = None  # pyserial Serial

# =========================
# Public API
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    cfg = {
      "port": "/dev/ttyACM0",
      "baud": 250000,
      "enabled": False,
      "homing_sequence": ["G28"],
      "feedrates": {"xy": 3000, "z": 600},
      "grid": {
          "origin_mm": [x0, y0],
          "pitch_mm": [dx, dy],
          "rows": R, "cols": C,
          "z": {"pick": -2.0, "travel": 15.0, "safe": 20.0}
      }
    }
    """
    global _state
    _state.port = str(cfg.get("port", _state.port))
    _state.baud = int(cfg.get("baud", _state.baud))
    _state.enabled = bool(cfg.get("enabled", _state.enabled))

    fr = cfg.get("feedrates", {}) or {}
    _state.feed_xy = int(fr.get("xy", _state.feed_xy))
    _state.feed_z = int(fr.get("z", _state.feed_z))

    g = cfg.get("grid", {}) or {}
    z = g.get("z", {}) or {}
    _state.grid = GridConfig(
        origin_mm=tuple(g.get("origin_mm", _state.grid.origin_mm)),   # type: ignore[arg-type]
        pitch_mm=tuple(g.get("pitch_mm", _state.grid.pitch_mm)),      # type: ignore[arg-type]
        rows=int(g.get("rows", _state.grid.rows)),
        cols=int(g.get("cols", _state.grid.cols)),
        z_pick=float(z.get("pick", _state.grid.z_pick)),
        z_travel=float(z.get("travel", _state.grid.z_travel)),
        z_safe=float(z.get("safe", _state.grid.z_safe)),
    )

    logger.info(
        "Motion init: port=%s baud=%s enabled=%s feed_xy=%s feed_z=%s grid=%s",
        _state.port, _state.baud, _state.enabled, _state.feed_xy, _state.feed_z, _state.grid
    )

    if _state.enabled:
        _open_serial()
        _run_homing()


def shutdown() -> None:
    _close_serial()
    logger.info("Motion shutdown complete")


def enabled() -> bool:
    return _state.enabled


def set_enabled(value: bool) -> bool:
    """Turn motion I/O on/off. Planning works regardless; execution requires enabled=True."""
    was = _state.enabled
    if value and not was:
        _state.enabled = True
        _open_serial()
        _run_homing()
    elif not value and was:
        _state.enabled = False
        _close_serial()
    return _state.enabled


def port_or_none() -> Optional[str]:
    return _state.port if _state.enabled else None


def plan_for(assignment_id: int) -> List[str]:
    """
    Build G-code for a given assignment.
    Expects models.get_assignment(assignment_id) to return a dict like:
      {"slot_id": 7, "row": 0, "col": 6, "x_mm": 140.0, "y_mm": 95.0, ...}
    Only one of (x_mm,y_mm), (row,col), or slot_id is required; we resolve the rest.
    """
    from app import models  # local import to avoid circulars
    asg = models.get_assignment(assignment_id)
    if not asg:
        raise ValueError(f"Assignment {assignment_id} not found")

    slot = _normalize_slot(asg)
    pose = _resolve_slot_pose(slot)  # (x_mm, y_mm, z_pick, z_travel, z_safe)

    g = _gcode_to_slot(pose)
    logger.debug("Planned G-code for assignment %s: %s", assignment_id, g)
    return g


def execute(move_id: int) -> Dict[str, Any]:
    """
    Execute a queued move (gcode list) by move_id using models.get_move(move_id).
    Requires motion to be enabled.
    """
    if not _state.enabled:
        return {"ok": False, "error": "Motion is disabled"}

    from app import models
    row = models.get_move(move_id)
    if not row:
        return {"ok": False, "error": f"Move {move_id} not found"}

    gcode: List[str] = row["gcode"]
    try:
        _send_gcode(gcode)
        return {"ok": True, "sent": len(gcode)}
    except Exception as e:
        logger.exception("Execute failed")
        return {"ok": False, "error": str(e)}

# =========================
# Slot / Pose resolution
# =========================

def _normalize_slot(asg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept a variety of assignment payloads and return a slot dict with any of:
      - 'x_mm','y_mm'
      - 'row','col'
      - 'slot_id'
    """
    # Already has explicit coordinates
    if "x_mm" in asg and "y_mm" in asg:
        return {"x_mm": float(asg["x_mm"]), "y_mm": float(asg["y_mm"])}

    # Has row/col
    if "row" in asg and "col" in asg:
        return {"row": int(asg["row"]), "col": int(asg["col"])}

    # Only a slot_id (typical for first_letter / csv lookup)
    if "slot_id" in asg and asg["slot_id"] is not None:
        return {"slot_id": int(asg["slot_id"])}

    # Try models.get_slot(slot_id) if assignment points to a slot row id
    sid = asg.get("slot_id")
    if sid is not None:
        return {"slot_id": int(sid)}

    raise ValueError("Assignment does not contain slot information")


def _resolve_slot_pose(slot: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """
    Resolve a slot dict into a physical pose (x_mm, y_mm, z_pick, z_travel, z_safe).
    Accepts any of:
      - {'x_mm': X, 'y_mm': Y}
      - {'row': r, 'col': c} (0-based)
      - {'slot_id': N} (1-based, row-major)
    """
    g = _state.grid

    # Direct XY
    if "x_mm" in slot and "y_mm" in slot:
        x = float(slot["x_mm"])
        y = float(slot["y_mm"])
        return (x, y, g.z_pick, g.z_travel, g.z_safe)

    # Row/Col (0-based)
    if "row" in slot and "col" in slot:
        r = int(slot["row"])
        c = int(slot["col"])
        if not (0 <= r < g.rows and 0 <= c < g.cols):
            raise ValueError(f"row/col out of range: ({r},{c}) with grid {g.rows}x{g.cols}")
        x = g.origin_mm[0] + c * g.pitch_mm[0]
        y = g.origin_mm[1] + r * g.pitch_mm[1]
        return (x, y, g.z_pick, g.z_travel, g.z_safe)

    # Linear slot_id (1-based)
    if "slot_id" in slot:
        sid = int(slot["slot_id"])
        total = g.rows * g.cols
        if not (1 <= sid <= total):
            raise ValueError(f"slot_id {sid} out of range 1..{total}")
        sid0 = sid - 1  # zero-based
        r = sid0 // g.cols
        c = sid0 % g.cols
        x = g.origin_mm[0] + c * g.pitch_mm[0]
        y = g.origin_mm[1] + r * g.pitch_mm[1]
        return (x, y, g.z_pick, g.z_travel, g.z_safe)

    raise ValueError("Slot must provide either (x_mm,y_mm), (row,col), or (slot_id)")


# =========================
# G-code generation
# =========================

def _gcode_to_slot(pose: Tuple[float, float, float, float, float]) -> List[str]:
    """
    Produce a conservative pick/place sequence to the slot pose.
    Pose: (x_mm, y_mm, z_pick, z_travel, z_safe)
    """
    x, y, z_pick, z_travel, z_safe = pose
    fxy = _state.feed_xy
    fz = _state.feed_z

    g: List[str] = []
    g.append("G90")                                       # absolute positioning
    g.append(f"G0 Z{z_safe:.3f} F{fz}")                   # safe Z
    g.append(f"G0 X{x:.3f} Y{y:.3f} F{fxy}")              # travel XY
    g.append(f"G1 Z{z_travel:.3f} F{fz}")                 # approach
    g.append(f"G1 Z{z_pick:.3f} F{fz}")                   # pick/deposit level
    g.append(f"G0 Z{z_safe:.3f} F{fz}")                   # retract
    return g


# =========================
# Serial I/O (optional)
# =========================

def _open_serial() -> None:
    global _ser
    try:
        import serial  # pyserial
    except ImportError:
        logger.error("pyserial not installed; motion cannot open serial")
        _state.enabled = False
        return
    try:
        _ser = serial.Serial(_state.port, _state.baud, timeout=1)
        time.sleep(2.0)  # allow controller to reset
        logger.info("Serial opened: %s @ %s", _state.port, _state.baud)
    except Exception as e:
        logger.error("Failed to open serial: %s", e, exc_info=True)
        _ser = None
        _state.enabled = False

def _close_serial() -> None:
    global _ser
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            logger.warning("Error closing serial", exc_info=True)
        _ser = None

def _run_homing() -> None:
    if _ser is None:
        return
    seq = _state.homing_sequence or []
    if not seq:
        return
    logger.info("Running homing sequence: %s", seq)
    _send_gcode(seq)

def _send_gcode(lines: List[str]) -> None:
    if _ser is None:
        raise RuntimeError("Serial not open")
    for line in lines:
        cmd = (line.strip() + "\n").encode()
        _ser.write(cmd)
        _ser.flush()
        # naive 'ok' wait; adjust per firmware if needed
        _read_until_ok()

def _read_until_ok() -> None:
    if _ser is None:
        return
    deadline = time.time() + 10.0
    buf = b""
    while time.time() < deadline:
        b = _ser.readline()
        if not b:
            continue
        buf += b
        s = buf.decode(errors="ignore").lower()
        if "ok" in s or "done" in s:
            return
    logger.warning("Timed out waiting for 'ok' from controller")
