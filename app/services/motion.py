# app/services/motion.py
from __future__ import annotations

import glob
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import serial  # pyserial
import serial.tools.list_ports

logger = logging.getLogger("motion")


# =============================================================================
# Configuration model
# =============================================================================

@dataclass
class MotionConfig:
    # Basics
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False

    # Serial behavior
    read_timeout_s: float = 2.0
    write_timeout_s: float = 2.0
    connect_timeout_s: float = 6.0
    reset_on_connect: bool = True          # toggle DTR/RTS to reset board
    startup_drain_s: float = 2.0           # drain boot banner after open

    # Protocol parsing (Marlin-like)
    ok_tokens: List[str] = field(default_factory=lambda: ["ok"])
    error_tokens: List[str] = field(default_factory=lambda: ["error", "Error:"])
    busy_tokens: List[str] = field(default_factory=lambda: ["busy:", "wait"])

    # Convenience motion params
    absolute_mode: bool = True
    safe_z_mm: float = 5.0
    travel_feed_xy: int = 1800
    travel_feed_z: int = 600

    home_on_start: bool = False
    home_axes: str = "XY"

    standby_pos: Optional[Dict[str, float]] = None  # {"x":..,"y":..,"z":..}
    test_square: Dict[str, Any] = field(default_factory=lambda: {"size_mm": 20.0, "feed_xy": 1200})

    # Robust auto-connect selectors
    by_id_contains: Optional[str] = None   # substring of /dev/serial/by-id/* symlink
    vid: Optional[str] = None              # hex string like "0483"
    pid: Optional[str] = None              # hex string like "5740"


# =============================================================================
# Module state
# =============================================================================

_cfg = MotionConfig()
_initialized = False
_enabled = False

_ser: Optional[serial.Serial] = None
_connected = False
_lock = threading.RLock()


# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_initialized():
    if not _initialized:
        raise RuntimeError("Motion service not initialized. Call motion.init() at startup.")

def _ensure_ready():
    if not _enabled:
        raise RuntimeError("Motion disabled. Call /motion/enable first.")
    if not is_connected():
        raise RuntimeError("Motion not connected. Call /motion/connect first.")

def _drain_startup(ser: serial.Serial, seconds: float):
    """
    Drain banner / residual input for a short period.
    """
    end = time.time() + max(0.0, seconds)
    prev_to = ser.timeout
    try:
        ser.timeout = 0.1
        while time.time() < end:
            try:
                pending = ser.in_waiting
            except Exception:
                pending = 0
            try:
                data = ser.read(pending or 1)
                if not data:
                    time.sleep(0.02)
                    continue
            except Exception:
                break
    finally:
        ser.timeout = prev_to


# =============================================================================
# Public API
# =============================================================================

def init(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize from config.yaml. Does NOT open the port.
    """
    global _cfg, _initialized, _enabled
    _cfg = MotionConfig(
        port=str(cfg.get("port", _cfg.port)),
        baud=int(cfg.get("baud", _cfg.baud)),
        enabled=bool(cfg.get("enabled", _cfg.enabled)),
        read_timeout_s=float(cfg.get("read_timeout_s", _cfg.read_timeout_s)),
        write_timeout_s=float(cfg.get("write_timeout_s", _cfg.write_timeout_s)),
        connect_timeout_s=float(cfg.get("connect_timeout_s", _cfg.connect_timeout_s)),
        reset_on_connect=bool(cfg.get("reset_on_connect", _cfg.reset_on_connect)),
        startup_drain_s=float(cfg.get("startup_drain_s", _cfg.startup_drain_s)),
        ok_tokens=list(cfg.get("ok_tokens", _cfg.ok_tokens)),
        error_tokens=list(cfg.get("error_tokens", _cfg.error_tokens)),
        busy_tokens=list(cfg.get("busy_tokens", _cfg.busy_tokens)),
        absolute_mode=bool(cfg.get("absolute_mode", _cfg.absolute_mode)),
        safe_z_mm=float(cfg.get("safe_z_mm", _cfg.safe_z_mm)),
        travel_feed_xy=int(cfg.get("travel_feed_xy", _cfg.travel_feed_xy)),
        travel_feed_z=int(cfg.get("travel_feed_z", _cfg.travel_feed_z)),
        home_on_start=bool(cfg.get("home_on_start", _cfg.home_on_start)),
        home_axes=str(cfg.get("home_axes", _cfg.home_axes)),
        standby_pos=cfg.get("standby_pos", _cfg.standby_pos),
        test_square=cfg.get("test_square", _cfg.test_square),
        by_id_contains=(str(cfg["by_id_contains"]).strip() if cfg.get("by_id_contains") else None),
        vid=(str(cfg["vid"]).lower().strip() if cfg.get("vid") else None),
        pid=(str(cfg["pid"]).lower().strip() if cfg.get("pid") else None),
    )
    _initialized = True
    _enabled = _cfg.enabled
    logger.info("Motion initialized with: %s", _cfg)
    return {"status": "initialized", "enabled": _enabled, "port": _cfg.port, "baud": _cfg.baud}


def list_ports() -> List[Dict[str, str]]:
    """
    Return a LIST of serial ports (rich metadata). This is the standardized shape.
    """
    ports: List[Dict[str, str]] = []
    for p in serial.tools.list_ports.comports():
        by_id = ""
        try:
            for link in glob.glob("/dev/serial/by-id/*"):
                if os.path.realpath(link) == p.device:
                    by_id = link
                    break
        except Exception:
            pass
        ports.append({
            "device": p.device,
            "by_id": by_id,
            "name": p.name or "",
            "description": p.description or "",
            "manufacturer": getattr(p, "manufacturer", "") or "",
            "product": getattr(p, "product", "") or "",
            "serial_number": getattr(p, "serial_number", "") or "",
            "hwid": p.hwid or "",
            "vid": f"{p.vid:04x}" if p.vid is not None else "",
            "pid": f"{p.pid:04x}" if p.pid is not None else "",
        })
    return ports


def is_connected() -> bool:
    with _lock:
        return bool(_connected and _ser is not None and _ser.is_open)


def enable() -> Dict[str, Any]:
    _ensure_initialized()
    global _enabled
    _enabled = True
    return {"status": "enabled"}

def disable() -> Dict[str, Any]:
    _ensure_initialized()
    global _enabled
    _enabled = False
    return {"status": "disabled"}

def close() -> Dict[str, Any]:
    """Close the serial port if open."""
    global _ser, _connected
    with _lock:
        try:
            if _ser and _ser.is_open:
                _ser.close()
        finally:
            _ser = None
            _connected = False
    return {"status": "closed"}


def connect(port: Optional[str] = None, baud: Optional[int] = None) -> Dict[str, Any]:
    """
    Open the serial port and drain the startup banner.
    """
    _ensure_initialized()
    global _ser, _connected

    with _lock:
        # Close any existing
        if _ser:
            try:
                _ser.close()
            except Exception:
                pass
            _ser = None
            _connected = False

        use_port = port or _cfg.port
        use_baud = int(baud or _cfg.baud)

        logger.info("Opening serial port %s @ %d", use_port, use_baud)
        ser = serial.Serial(
            port=use_port,
            baudrate=use_baud,
            timeout=_cfg.read_timeout_s,
            write_timeout=_cfg.write_timeout_s,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )

        # Reset pulse (if supported)
        if _cfg.reset_on_connect:
            try:
                ser.dtr = False
                ser.rts = False
                time.sleep(0.05)
                ser.dtr = True
                ser.rts = True
            except Exception:
                logger.debug("DTR/RTS toggle not supported", exc_info=True)

        _drain_startup(ser, _cfg.startup_drain_s)

        _ser = ser
        _connected = True

        if _cfg.home_on_start:
            try:
                home(_cfg.home_axes)
            except Exception as e:
                logger.warning("Home-on-start failed: %s", e)

    return {"status": "connected", "port": use_port, "baud": use_baud}


def auto_connect() -> Dict[str, Any]:
    """
    Try to auto-select a port using:
      1) by-id substring
      2) VID/PID exact match
      3) configured port
      4) only available port
    """
    _ensure_initialized()
    ports = list_ports()
    if not ports:
        return {"connected": False, "port": "", "reason": "no ports present"}

    # 1) by-id substring
    if _cfg.by_id_contains:
        needle = _cfg.by_id_contains.lower()
        for p in ports:
            if needle in (p.get("by_id") or "").lower():
                res = connect(port=p["by_id"] or p["device"], baud=_cfg.baud)
                return {"connected": True, "port": res["port"], "reason": "matched by by-id"}

    # 2) VID/PID exact
    if _cfg.vid and _cfg.pid:
        vid = _cfg.vid.lower()
        pid = _cfg.pid.lower()
        for p in ports:
            if p.get("vid", "").lower() == vid and p.get("pid", "").lower() == pid:
                res = connect(port=p["device"], baud=_cfg.baud)
                return {"connected": True, "port": res["port"], "reason": "matched by vid:pid"}

    # 3) Configured device
    if _cfg.port:
        for p in ports:
            if p.get("device") == _cfg.port:
                res = connect(port=_cfg.port, baud=_cfg.baud)
                return {"connected": True, "port": res["port"], "reason": "matched configured port"}

    # 4) Only one port
    if len(ports) == 1:
        p = ports[0]
        res = connect(port=p["by_id"] or p["device"], baud=_cfg.baud)
        return {"connected": True, "port": res["port"], "reason": "single available port"}

    return {"connected": False, "port": "", "reason": "multiple ports; no unique match"}


# =============================================================================
# Send / receive
# =============================================================================

def _write_line(ser: serial.Serial, line: str):
    data = (line.strip() + "\n").encode("ascii", errors="replace")
    ser.write(data)
    ser.flush()

def _read_until_ok_or_error(ser: serial.Serial, per_cmd_timeout_s: float = None) -> List[str]:
    """
    Read lines until OK or Error appears. Busy/info lines are collected and ignored.
    """
    lines: List[str] = []
    deadline = time.time() + (per_cmd_timeout_s if per_cmd_timeout_s is not None else max(1.0, _cfg.connect_timeout_s))

    while True:
        if time.time() > deadline:
            raise TimeoutError("Timeout waiting for firmware response (no 'ok' or 'error').")

        raw = ser.readline()
        if not raw:
            continue

        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue

        lines.append(text)
        low = text.lower()

        if any(tok in low for tok in map(str.lower, _cfg.error_tokens)):
            raise RuntimeError(f"Firmware reported error: {text}")

        if any(tok in low for tok in map(str.lower, _cfg.busy_tokens)):
            continue

        if any(tok in low for tok in map(str.lower, _cfg.ok_tokens)):
            break

    return lines

def send(gcode_lines: List[str]) -> Dict[str, Any]:
    """
    Send one or more G-code lines, waiting for 'ok' after each.
    """
    _ensure_ready()
    if not gcode_lines:
        raise ValueError("No G-code provided.")

    move_id = str(uuid.uuid4())
    responses: List[Dict[str, Any]] = []

    with _lock:
        assert _ser is not None and _ser.is_open
        for ln in [l for l in gcode_lines if l and l.strip()]:
            _write_line(_ser, ln)
            lines = _read_until_ok_or_error(_ser)
            responses.append({"line": ln.strip(), "reply": lines})

    return {"move_id": move_id, "lines": responses}


# =============================================================================
# Convenience commands
# =============================================================================

def ping() -> Dict[str, Any]:
    """Query firmware info (Marlin: M115)."""
    return send(["M115"])

def home(axes: str = "XY") -> Dict[str, Any]:
    """
    Home the given axes. Example: axes='XY' or 'XYZ'.
    Marlin: 'G28 X Y' homes X and Y; 'G28' homes all.
    """
    axes = (axes or "").upper()
    cmd = "G28" if axes in ("", "XYZ") else f"G28 {' '.join(list(axes))}"
    return send([cmd])

def test_square(size_mm: float = None, feed_xy: int = None) -> Dict[str, Any]:
    """
    Draw a square in the XY plane (relative mode) starting at current position.
    """
    size = float(size_mm if size_mm is not None else _cfg.test_square.get("size_mm", 20.0))
    feed = int(feed_xy if feed_xy is not None else _cfg.test_square.get("feed_xy", 1200))
    seq = [
        "G91",
        f"G0 X{size:.3f} F{feed}",
        f"G0 Y{size:.3f} F{feed}",
        f"G0 X{-size:.3f} F{feed}",
        f"G0 Y{-size:.3f} F{feed}",
        "G90",
    ]
    return send(seq)

def jog(axis: str, delta_mm: float, feed_xy: int = 1200) -> Dict[str, Any]:
    """Relative jog on a single axis, then return to absolute."""
    axis = (axis or "").upper().strip()
    if axis not in ("X", "Y", "Z"):
        raise ValueError("axis must be one of X, Y, Z")
    d = float(delta_mm); f = int(feed_xy)
    return send(["G91", f"G0 {axis}{d:.3f} F{f}", "G90"])

def goto_xy(x_mm: float, y_mm: float, safe_lift: bool = True) -> Dict[str, Any]:
    """Absolute XY travel with optional safe Z lift first."""
    _ensure_ready()
    seq = ["G90"]
    if safe_lift and _cfg.safe_z_mm is not None:
        seq.append(f"G0 Z{float(_cfg.safe_z_mm):.3f} F{_cfg.travel_feed_z}")
    seq.append(f"G0 X{float(x_mm):.3f} Y{float(y_mm):.3f} F{_cfg.travel_feed_xy}")
    return send(seq)

def go_standby() -> Optional[Dict[str, Any]]:
    """Go to the configured standby position (if provided)."""
    if not _cfg.standby_pos:
        return None
    x = _cfg.standby_pos.get("x")
    y = _cfg.standby_pos.get("y")
    z = _cfg.standby_pos.get("z")
    parts = ["G90"]
    if z is not None:
        parts.append(f"G0 Z{float(z):.3f} F{_cfg.travel_feed_z}")
    if x is not None or y is not None:
        gx = f"X{float(x):.3f}" if x is not None else ""
        gy = f"Y{float(y):.3f}" if y is not None else ""
        parts.append(f"G0 {gx} {gy} F{_cfg.travel_feed_xy}".strip())
    return send(parts)
