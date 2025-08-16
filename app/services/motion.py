# app/services/motion.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import serial  # pyserial
import serial.tools.list_ports

logger = logging.getLogger("motion")

# =========
# Config
# =========

@dataclass
class MotionConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False

    # Serial behavior
    read_timeout_s: float = 2.0
    write_timeout_s: float = 2.0
    connect_timeout_s: float = 6.0
    reset_on_connect: bool = True          # toggle DTR/RTS to reset the board
    startup_drain_s: float = 2.0           # time to drain boot banner after reset

    # Protocol parsing (Marlin-like)
    ok_tokens: tuple = ("ok",)             # lines containing these tokens signal success
    error_tokens: tuple = ("error", "Error:")  # signals failure
    busy_tokens: tuple = ("busy:", "wait")     # informational; do not fail

# =========
# State
# =========

_serial_lock = threading.RLock()
_ser: Optional[serial.Serial] = None
_cfg = MotionConfig()
_initialized = False

# =========
# Public API expected by main.py / UI
# =========

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize motion service using config.yaml values.
    Does not open the port yet; call connect()/enable() when ready.
    """
    global _cfg, _initialized
    _cfg = MotionConfig(
        port=str(cfg.get("port", _cfg.port)),
        baud=int(cfg.get("baud", _cfg.baud)),
        enabled=bool(cfg.get("enabled", _cfg.enabled)),
        read_timeout_s=float(cfg.get("read_timeout_s", _cfg.read_timeout_s)),
        write_timeout_s=float(cfg.get("write_timeout_s", _cfg.write_timeout_s)),
        connect_timeout_s=float(cfg.get("connect_timeout_s", _cfg.connect_timeout_s)),
        reset_on_connect=bool(cfg.get("reset_on_connect", _cfg.reset_on_connect)),
        startup_drain_s=float(cfg.get("startup_drain_s", _cfg.startup_drain_s)),
        ok_tokens=tuple(cfg.get("ok_tokens", list(_cfg.ok_tokens))),
        error_tokens=tuple(cfg.get("error_tokens", list(_cfg.error_tokens))),
        busy_tokens=tuple(cfg.get("busy_tokens", list(_cfg.busy_tokens))),
    )
    _initialized = True
    logger.info(
        "Motion init: port=%s baud=%d enabled=%s",
        _cfg.port, _cfg.baud, _cfg.enabled
    )

def status() -> bool:
    return bool(_initialized)

def list_ports() -> List[Dict[str, str]]:
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "device": p.device,
            "name": p.name or "",
            "description": p.description or "",
            "hwid": p.hwid or "",
            "vid": f"{p.vid:04x}" if p.vid is not None else "",
            "pid": f"{p.pid:04x}" if p.pid is not None else "",
        })
    return ports

def is_connected() -> bool:
    with _serial_lock:
        return _ser is not None and _ser.is_open

def connect(port: Optional[str] = None, baud: Optional[int] = None) -> Dict[str, Any]:
    """
    Open the serial port and drain the startup banner. Safe to call repeatedly.
    """
    global _ser
    prt = port or _cfg.port
    bd = baud or _cfg.baud

    with _serial_lock:
        # Already connected to the same port/baud
        if _ser and _ser.is_open and _ser.port == prt and _ser.baudrate == bd:
            return {"connected": True, "port": prt, "baud": bd, "already": True}

        # Close any existing
        try:
            if _ser and _ser.is_open:
                _ser.close()
        except Exception:
            pass
        _ser = None

        # Open new
        logger.info("Opening serial: %s @ %d", prt, bd)
        ser = serial.Serial()
        ser.port = prt
        ser.baudrate = bd
        ser.timeout = _cfg.read_timeout_s
        ser.write_timeout = _cfg.write_timeout_s
        ser.dsrdtr = False
        ser.rtscts = False
        ser.xonxoff = False

        ser.open()

        # Toggle DTR/RTS to reset board if requested
        if _cfg.reset_on_connect:
            try:
                ser.dtr = False
                ser.rts = False
                time.sleep(0.1)
                ser.dtr = True
                ser.rts = True
            except Exception:
                # Not all adapters support these lines; ignore
                logger.debug("DTR/RTS toggle not supported", exc_info=True)

        # Drain startup banner
        t0 = time.time()
        deadline = t0 + _cfg.connect_timeout_s
        time.sleep(_cfg.startup_drain_s)
        _drain_input(ser, deadline)

        _ser = ser
        return {"connected": True, "port": prt, "baud": bd, "already": False}

def disconnect() -> Dict[str, Any]:
    global _ser
    with _serial_lock:
        if _ser:
            try:
                _ser.close()
            except Exception:
                pass
        _ser = None
    return {"connected": False}

def enable() -> Dict[str, Any]:
    """
    Ensure connected and mark enabled. Does NOT send machine-enable G-code.
    """
    if not is_connected():
        connect()
    _cfg.enabled = True
    return {"enabled": True, "port": _cfg.port, "baud": _cfg.baud}

def disable() -> Dict[str, Any]:
    _cfg.enabled = False
    return {"enabled": False}

def ping() -> Dict[str, Any]:
    """
    Query firmware info (Marlin: M115). Returns raw response lines.
    """
    _ensure_ready()
    lines = send_and_wait(["M115"])
    return {"ok": True, "response": lines}

def home(axes: str = "XY") -> Dict[str, Any]:
    """
    Home axes (Marlin: G28). Example: axes='X', 'Y', 'XY', 'XYZ', or '' for all configured.
    """
    _ensure_ready()
    cmd = "G28" if not axes else f"G28 { ' '.join(list(axes)) }"
    send_and_wait([cmd])
    return {"ok": True}

def test_square(size_mm: float = 20.0, feed_xy: int = 1200) -> Dict[str, Any]:
    """
    Draw a simple square in absolute coords around current origin.
    """
    _ensure_ready()
    s = float(size_mm)
    f = int(feed_xy)

    g = [
        "M400",              # wait for any prior moves
        "G90",               # absolute mode
        f"G0 Z10 F{f}",      # safe Z
        "G92 X0 Y0",         # set current as (0,0)
        f"G0 X0 Y0 F{f}",
        f"G1 X{s} Y0 F{f}",
        f"G1 X{s} Y{s} F{f}",
        f"G1 X0 Y{s} F{f}",
        f"G1 X0 Y0 F{f}",
        "M400"
    ]
    send_and_wait(g)
    return {"ok": True, "ran": len(g)}

def send(gcode: List[str]) -> Dict[str, Any]:
    """
    Send raw G-code lines and wait for their 'ok' acknowledgements.
    """
    _ensure_ready()
    lines = [ln.strip() for ln in gcode if ln and ln.strip()]
    if not lines:
        return {"ok": True, "count": 0, "response": []}
    resp = send_and_wait(lines)
    return {"ok": True, "count": len(lines), "response": resp}

# =========
# Core serial helpers
# =========

def _ensure_ready() -> None:
    if not _initialized:
        raise RuntimeError("Motion not initialized")
    if not _cfg.enabled:
        raise RuntimeError("Motion disabled (enable first)")
    if not is_connected():
        connect()

def _drain_input(ser: serial.Serial, deadline: float) -> List[str]:
    """
    Read and discard any startup banner / residual lines until timeout.
    """
    seen: List[str] = []
    while time.time() < deadline:
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            break
        try:
            line = raw.decode(errors="ignore").strip()
        except Exception:
            line = ""
        if line:
            seen.append(line)
    if seen:
        logger.debug("Startup drain:\n%s", "\n".join(seen))
    return seen

def _write_line(ser: serial.Serial, line: str) -> None:
    data = (line.strip() + "\n").encode("ascii", errors="ignore")
    ser.write(data)
    ser.flush()

def send_and_wait(lines: List[str], overall_timeout_s: float = 60.0) -> List[str]:
    """
    Send lines, wait for an 'ok' (or 'error') per line. Collect responses.
    Returns all non-empty response lines (including 'ok' tokens).
    """
    with _serial_lock:
        if not _ser or not _ser.is_open:
            raise RuntimeError("Serial not open")

        ser = _ser
        responses: List[str] = []
        t_stop = time.time() + overall_timeout_s

        for idx, ln in enumerate(lines, start=1):
            _write_line(ser, ln)
            logger.debug(">> %s", ln)

            # Read until an ok/error appears for this line
            while True:
                if time.time() > t_stop:
                    raise TimeoutError(f"Timeout waiting for 'ok' after sending: {ln!r}")

                raw = ser.readline()
                if not raw:
                    # pyserial timeout hit for this read; keep looping until overall timeout
                    continue

                try:
                    text = raw.decode(errors="ignore").strip()
                except Exception:
                    text = ""

                if not text:
                    continue

                logger.debug("<< %s", text)
                responses.append(text)

                low = text.lower()
                # Error?
                if any(tok in low for tok in map(str.lower, _cfg.error_tokens)):
                    raise RuntimeError(f"Firmware reported error after '{ln}': {text}")

                # Busy/info lines — continue reading
                if any(tok in low for tok in map(str.lower, _cfg.busy_tokens)):
                    continue

                # OK?
                if any(tok in low for tok in map(str.lower, _cfg.ok_tokens)):
                    break  # proceed to next command

        return responses
