# app/services/plunger.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

logger = logging.getLogger("plunger")

# -------------------------
# Runtime / backend state
# -------------------------
_BACKEND: Optional[Literal["pigpio", "rpigpio"]] = None
_LOCK = threading.RLock()

# pigpio runtime
_pigpio = None  # module
_pi = None      # pigpio.pi()

# RPi.GPIO runtime
_rpi = None     # module

# -------------------------
# Configuration
# -------------------------
@dataclass
class Pins:
    enable_pin: int = 18          # BCM numbering
    step_pin: int = 23
    dir_pin: int = 24
    enable_active_low: bool = True
    dir_invert: bool = False

@dataclass
class State:
    pins: Pins = field(default_factory=Pins)
    initialized: bool = False
    enabled: bool = False

_state = State()

_MIN_DELAY_US = 200               # half-period floor (us)
_MAX_STEPS = 500_000              # safety cap per jog

# -------------------------
# Public API
# -------------------------
def init(cfg: Optional[Dict] = None) -> None:
    """
    Initialize plunger backend.
    Optional cfg keys:
      enable_pin, step_pin, dir_pin (BCM), enable_active_low, dir_invert,
      backend: "pigpio" | "rpigpio" | None (auto)
    """
    global _BACKEND, _pigpio, _pi, _rpi

    with _LOCK:
        if _state.initialized:
            logger.info("plunger.init(): already initialized (backend=%s)", _BACKEND)
            return

        # Apply config
        forced = None
        if cfg:
            p = _state.pins
            _state.pins = Pins(
                enable_pin=int(cfg.get("enable_pin", p.enable_pin)),
                step_pin=int(cfg.get("step_pin", p.step_pin)),
                dir_pin=int(cfg.get("dir_pin", p.dir_pin)),
                enable_active_low=bool(cfg.get("enable_active_low", p.enable_active_low)),
                dir_invert=bool(cfg.get("dir_invert", p.dir_invert)),
            )
            forced = cfg.get("backend")

        # Backend choice
        if forced in ("pigpio", "rpigpio"):
            choice = forced
        else:
            # Prefer pigpio if daemon running
            try:
                import pigpio as _tmp_pigpio  # type: ignore
                pi = _tmp_pigpio.pi()
                if not pi.connected:
                    raise RuntimeError("pigpio daemon not running")
                _pigpio = _tmp_pigpio
                _pi = pi
                choice = "pigpio"
            except Exception as e:
                logger.info("pigpio unavailable (%s); will try RPi.GPIO", e)
                choice = "rpigpio"

        if choice == "pigpio":
            try:
                # Set mode
                _pi.set_mode(_state.pins.enable_pin, _pigpio.OUTPUT)
                _pi.set_mode(_state.pins.step_pin, _pigpio.OUTPUT)
                _pi.set_mode(_state.pins.dir_pin, _pigpio.OUTPUT)
                # Mark backend BEFORE touching helpers that depend on it
                _set_backend("pigpio")
                # Initialize pins
                _pi.write(_state.pins.step_pin, 0)
                _set_enable_hw(False)
                _state.initialized = True
                logger.info(
                    "Plunger initialized (pigpio) EN=%s STEP=%s DIR=%s",
                    _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin
                )
                return
            except Exception as e:
                logger.warning("pigpio init failed: %s", e, exc_info=True)
                # fall through to try RPi.GPIO

        # Fallback: RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore
            _rpi = GPIO
            _rpi.setmode(GPIO.BCM)
            _rpi.setwarnings(False)
            # Mark backend BEFORE helper use
            _set_backend("rpigpio")
            # Configure pins
            _rpi.setup(_state.pins.enable_pin, GPIO.OUT, initial=_inactive_level())
            _rpi.setup(_state.pins.step_pin, GPIO.OUT, initial=GPIO.LOW)
            _rpi.setup(_state.pins.dir_pin, GPIO.OUT, initial=GPIO.LOW)
            _state.initialized = True
            logger.info(
                "Plunger initialized (RPi.GPIO) EN=%s STEP=%s DIR=%s",
                _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin
            )
        except Exception as e:
            # Soft-fail: do not crash the whole app
            _clear_backend()
            _state.initialized = False
            logger.warning(
                "No usable GPIO backend (install/start pigpio OR install python3-rpi.gpio). "
                "Plunger disabled; server will still start. Details: %s", e
            )
            return

def jog(steps: int, enable: bool = True, delay_us: int = 1200) -> Dict[str, object]:
    """Jog stepper by a signed number of steps."""
    if not _state.initialized or _BACKEND is None:
        raise RuntimeError("Plunger not initialized (no GPIO backend)")

    if steps == 0:
        return {"ok": True, "steps": 0, "direction": "none", "backend": _BACKEND}

    if abs(steps) > _MAX_STEPS:
        raise ValueError(f"steps exceeds safety limit ({_MAX_STEPS})")

    delay_us = max(_MIN_DELAY_US, int(delay_us))

    direction_fwd = (steps > 0)
    if _state.pins.dir_invert:
        direction_fwd = not direction_fwd
    abs_steps = abs(steps)

    with _LOCK:
        prev_enabled = _state.enabled
        if enable:
            _set_enable_hw(True)

        _set_dir_hw(direction_fwd)

        if _BACKEND == "pigpio":
            _pulse_pigpio(abs_steps, delay_us)
        elif _BACKEND == "rpigpio":
            _pulse_rpigpio(abs_steps, delay_us)
        else:
            raise RuntimeError("No GPIO backend available")

        if enable and not prev_enabled:
            _set_enable_hw(False)

    return {
        "ok": True,
        "steps": steps,
        "direction": "forward" if steps > 0 else "reverse",
        "backend": _BACKEND,
        "delay_us": delay_us,
    }

def shutdown() -> None:
    """Disable driver and release resources."""
    with _LOCK:
        try:
            if _state.initialized and _BACKEND is not None:
                _set_enable_hw(False)
        except Exception:
            logger.warning("Error disabling driver on shutdown", exc_info=True)

        # Stop pigpio
        try:
            if _BACKEND == "pigpio" and _pi is not None:
                _pi.stop()
        except Exception:
            logger.warning("Error stopping pigpio", exc_info=True)

        # Cleanup RPi.GPIO
        try:
            if _BACKEND == "rpigpio" and _rpi is not None:
                _rpi.cleanup()
        except Exception:
            logger.warning("RPi.GPIO cleanup failed", exc_info=True)

        _clear_backend()
        _state.initialized = False
        _state.enabled = False
        logger.info("Plunger shutdown complete")

# -------------------------
# Helpers
# -------------------------
def _set_backend(name: Optional[str]) -> None:
    global _BACKEND
    if name not in (None, "pigpio", "rpigpio"):
        raise ValueError("invalid backend")
    _BACKEND = name  # type: ignore

def _clear_backend() -> None:
    global _pi, _pigpio, _rpi, _BACKEND
    _pi = None
    _pigpio = None
    _rpi = None
    _BACKEND = None

def _inactive_level() -> int:
    return 0 if not _state.pins.enable_active_low else 1

def _active_level() -> int:
    return 1 - _inactive_level()

def _set_enable_hw(value: bool) -> None:
    if _BACKEND == "pigpio":
        lvl = _active_level() if value else _inactive_level()
        _pi.write(_state.pins.enable_pin, lvl)
    elif _BACKEND == "rpigpio":
        lvl = _active_level() if value else _inactive_level()
        _rpi.output(_state.pins.enable_pin, lvl)
    else:
        raise RuntimeError("No GPIO backend available")
    _state.enabled = value

def _set_dir_hw(forward: bool) -> None:
    if _BACKEND == "pigpio":
        _pi.write(_state.pins.dir_pin, 1 if forward else 0)
    elif _BACKEND == "rpigpio":
        _rpi.output(_state.pins.dir_pin, 1 if forward else 0)
    else:
        raise RuntimeError("No GPIO backend available")

def _pulse_pigpio(steps: int, delay_us: int) -> None:
    step_gpio = _state.pins.step_pin
    pulses = []
    for _ in range(steps):
        pulses.append(_pigpio.pulse(1 << step_gpio, 0, delay_us))
        pulses.append(_pigpio.pulse(0, 1 << step_gpio, delay_us))
    _pi.wave_clear()
    _pi.wave_add_generic(pulses)
    wave_id = _pi.wave_create()
    if wave_id < 0:
        raise RuntimeError("pigpio wave_create failed")
    try:
        _pi.wave_send_once(wave_id)
        while _pi.wave_tx_busy():
            time.sleep(0.001)
    finally:
        _pi.wave_delete(wave_id)

def _pulse_rpigpio(steps: int, delay_us: int) -> None:
    pin = _state.pins.step_pin
    on_t = delay_us / 1_000_000.0
    off_t = on_t
    for _ in range(steps):
        _rpi.output(pin, _rpi.HIGH)
        time.sleep(on_t)
        _rpi.output(pin, _rpi.LOW)
        time.sleep(off_t)
