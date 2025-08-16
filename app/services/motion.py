# app/services/motion.py
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import serial  # pyserial
import serial.tools.list_ports

logger = logging.getLogger("motion")


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class MotionConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False

    read_timeout_s: float = 2.0
    write_timeout_s: float = 2.0
    connect_timeout_s: float = 6.0

    reset_on_connect: bool = True
    startup_drain_s: float = 2.0

    ok_tokens: List[str] = None
    error_tokens: List[str] = None
    busy_tokens: List[str] = None

    absolute_mode: bool = True
    safe_z_mm: float = 5.0
    travel_feed_xy: int = 1800
    travel_feed_z: int = 600

    home_on_start: bool = False
    home_axes: str = "XY"

    standby_pos: Optional[Dict[str, float]] = None  # {"x":0.0,"y":0.0,"z":10.0}

    test_square: Dict[str, Any] = None  # {"size_mm":20.0,"feed_xy":1200}

    def __post_init__(self):
        if self.ok_tokens is None:
            self.ok_tokens = ["ok"]
        if self.error_tokens is None:
            self.error_tokens = ["error", "Error:"]
        if self.busy_tokens is None:
            self.busy_tokens = ["busy:", "wait"]
        if self.test_square is None:
            self.test_square = {"size_mm": 20.0, "feed_xy": 1200}


# =============================================================================
# Module global state
# =============================================================================

_cfg = MotionConfig()
_initialized = False

_ser: Optional[serial.Serial] = None
_lock = threading.RLock()
_connected = False
_enabled = False


# =============================================================================
# Helpers
# =============================================================================

def _ensure_initialized():
    if not _initialized:
        raise RuntimeError("Motion service not initialized. Call motion.init() at startup.")

def _ensure_ready():
    if not _enabled:
        raise RuntimeError("Motion disabled. Call /motion/enable first.")
    if not _connected or _ser is None or not _ser.is_open:
        raise RuntimeError("Motion not connected. Call /motion/connect first.")

def _drain_startup(ser: serial.Serial, seconds: float):
    """Read and discard any banner or buffered output right after opening the port."""
    end = time.time() + max(0.0, seconds)
    ser.timeout = 0.1
    while time.time() < end:
        try:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                time.sleep(0.02)
                continue
        except Exception:
            break


# =============================================================================
# Public API
# =============================================================================

def init(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize motion service from config.yaml values.
    This does not open the port; use connect()/enable() when ready.
    """
    global _cfg, _initialized, _enabled

    # Rebuild config while preserving defaults
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
    )

    _initialized = True
    _enabled = _cfg.enabled

    logger.info("Motion initialized with: %s", _cfg)
    return {"status": "initialized", "enabled": _enabled, "port": _cfg.port, "baud": _cfg.baud}


def list_ports() -> Dict[str, Any]:
    """Enumerate serial ports the host can see."""
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({"device": p.device, "name": p.name, "hwid": p.hwid})
    return {"ports": ports}


def connect(port: Optional[str] = None, baud: Optional[int] = None) -> Dict[str, Any]:
    """Open the serial port to the controller and drain the startup banner."""
    _ensure_initialized()
    global _ser, _connected

    with _lock:
        if _ser and _ser.is_open:
            try:
                _ser.close()
            except Exception:
                pass
            _ser = None
            _connected = False

        use_port = port or _cfg.port
        use_baud = baud or _cfg.baud

        logger.info("Opening serial port %s @ %s", use_port, use_baud)
        ser = serial.Serial(
            port=use_port,
            baudrate=use_baud,
            timeout=_cfg.read_timeout_s,
            write_timeout=_cfg.write_timeout_s,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )

        # Toggle DTR/RTS if requested (some boards reset on DTR)
        if _cfg.reset_on_connect:
            try:
                ser.dtr = False
                ser.rts = False
                time.sleep(0.05)
                ser.dtr = True
                ser.rts = True
            except Exception as e:
                logger.debug("DTR/RTS toggle unsupported: %s", e)

        # Drain banner / buffer
        _drain_startup(ser, _cfg.startup_drain_s)

        _ser = ser
        _connected = True

        # Optionally send a homing sequence once at connect
        if _cfg.home_on_start:
            try:
                home(_cfg.home_axes)
            except Exception as e:
                logger.warning("Home-on-start failed: %s", e)

    return {"status": "connected", "port": use_port, "baud": use_baud}


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


# =============================================================================
# Core send-and-wait logic
# =============================================================================

def _write_line(ser: serial.Serial, line: str):
    data = (line.strip() + "\n").encode("ascii", errors="replace")
    ser.write(data)
    ser.flush()

def _read_until_ok_or_error(ser: serial.Serial) -> List[str]:
    """
    Read response lines until we encounter an OK (success) or Error token.
    Busy/info tokens are logged and ignored while we continue waiting.
    """
    lines: List[str] = []
    deadline = time.time() + max(1.0, _cfg.connect_timeout_s)

    while True:
        if time.time() > deadline:
            raise TimeoutError("Timeout waiting for firmware response (no 'ok' or 'error').")

        raw = ser.readline()
        if not raw:
            continue

        text = raw.decode("utf-8", errors="replace").strip()
        if text:
            lines.append(text)
            low = text.lower()

            # Error?
            if any(tok in low for tok in map(str.lower, _cfg.error_tokens)):
                raise RuntimeError(f"Firmware reported error: {text}")

            # Busy/info — keep waiting
            if any(tok in low for tok in map(str.lower, _cfg.busy_tokens)):
                continue

            # OK?
            if any(tok in low for tok in map(str.lower, _cfg.ok_tokens)):
                break

    return lines


def send(gcode_lines: List[str]) -> Dict[str, Any]:
    """
    Send one or more G-code lines and wait for 'ok' after each line.
    Returns a move_id that the UI can display and a list of response lines.
    """
    _ensure_ready()
    if not gcode_lines:
        raise ValueError("No G-code provided.")

    move_id = str(uuid.uuid4())
    responses: List[Dict[str, Any]] = []

    with _lock:
        assert _ser is not None
        for idx, ln in enumerate(gcode_lines, start=1):
            line = ln.strip()
            if not line:
                continue
            _write_line(_ser, line)
            try:
                lines = _read_until_ok_or_error(_ser)
            except Exception as e:
                logger.error("Error after sending '%s': %s", line, e)
                raise
            responses.append({"line": line, "reply": lines})

    return {"move_id": move_id, "lines": responses}


# =============================================================================
# Convenience commands
# =============================================================================

def ping() -> Dict[str, Any]:
    """Query firmware capabilities (Marlin: M115)."""
    return send(["M115"])

def home(axes: str = "XY") -> Dict[str, Any]:
    """
    Home the given axes. Example: axes='XY' or 'XYZ'.
    On Marlin, 'G28 X Y' homes X and Y; 'G28' homes all.
    """
    axes = (axes or "").upper()
    cmd = "G28" if axes == "" or axes == "XYZ" else f"G28 {' '.join(list(axes))}"
    return send([cmd])

def test_square(size_mm: float = None, feed_xy: int = None) -> Dict[str, Any]:
    """
    Draw a small square in the XY plane, starting at current position.
    Uses relative mode to avoid dependency on work offsets.
    """
    size = float(size_mm if size_mm is not None else _cfg.test_square.get("size_mm", 20.0))
    feed = int(feed_xy if feed_xy is not None else _cfg.test_square.get("feed_xy", 1200))

    seq = [
        "G91",                        # relative
        f"G0 X{size:.3f} F{feed}",    # +X
        f"G0 Y{size:.3f} F{feed}",    # +Y
        f"G0 X{-size:.3f} F{feed}",   # -X
        f"G0 Y{-size:.3f} F{feed}",   # -Y
        "G90",                        # absolute
    ]
    return send(seq)

def jog(axis: str, delta_mm: float, feed_xy: int = 1200) -> Dict[str, Any]:
    """
    Relative jog on a single axis, then return to absolute mode.
    Example: jog('X', 10) → G91; G0 X10 F1200; G90
    """
    axis = (axis or "").upper().strip()
    if axis not in ("X", "Y", "Z"):
        raise ValueError("axis must be one of X, Y, Z")
    d = float(delta_mm)
    f = int(feed_xy)
    seq = ["G91", f"G0 {axis}{d:.3f} F{f}", "G90"]
    return send(seq)


# =============================================================================
# Optional helpers for planned motion (standby, safe Z)
# =============================================================================

def go_standby() -> Optional[Dict[str, Any]]:
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
