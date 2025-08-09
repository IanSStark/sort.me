# app/services/plunger.py
from __future__ import annotations
import logging, threading, time
from dataclasses import dataclass
from typing import Optional, Literal, Dict

logger = logging.getLogger("plunger")

_BACKEND: Optional[Literal["pigpio", "rpigpio"]] = None
_LOCK = threading.RLock()

# pigpio runtime
_pigpio = None  # module
_pi = None      # pigpio.pi()

# RPi.GPIO runtime
_rpi = None     # module

@dataclass
class Pins:
    enable_pin: int = 18   # BCM numbering
    step_pin: int = 23
    dir_pin: int = 24
    enable_active_low: bool = True
    dir_invert: bool = False

@dataclass
class State:
    pins: Pins = Pins()
    initialized: bool = False
    enabled: bool = False

_state = State()
_MIN_DELAY_US = 200
_MAX_STEPS = 500_000

def init(cfg: Optional[Dict] = None) -> None:
    """Initialize GPIO backend and pins."""
    global _BACKEND, _pigpio, _pi, _rpi
    with _LOCK:
        if _state.initialized:
            logger.info("plunger.init(): already initialized (backend=%s)", _BACKEND)
            return

        # apply cfg
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
        else:
            forced = None

        # choose backend
        if forced in ("pigpio", "rpigpio"):
            choice = forced
        else:
            # prefer pigpio if available and daemon running
            try:
                import pigpio as _tmp
                pi = _tmp.pi()
                if not pi.connected:
                    raise RuntimeError("pigpio daemon not running")
                _pigpio = _tmp
                _pi = pi
                choice = "pigpio"
            except Exception as e:
                logger.info("pigpio unavailable (%s); falling back to RPi.GPIO", e)
                choice = "rpigpio"

        if choice == "pigpio":
            _pi.set_mode(_state.pins.enable_pin, _pigpio.OUTPUT)
            _pi.set_mode(_state.pins.step_pin, _pigpio.OUTPUT)
            _pi.set_mode(_state.pins.dir_pin, _pigpio.OUTPUT)
            _set_enable_hw(False)
            _pi.write(_state.pins.step_pin, 0)
            _BACKEND = "pigpio"
            _state.initialized = True
            logger.info("Plunger ready (pigpio) EN=%s STEP=%s DIR=%s",
                        _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin)
            return

        # fallback RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore
            _rpi = GPIO
            _rpi.setmode(GPIO.BCM)
            _rpi.setwarnings(False)
            _rpi.setup(_state.pins.enable_pin, GPIO.OUT, initial=_inactive_level())
            _rpi.setup(_state.pins.step_pin, GPIO.OUT, initial=GPIO.LOW)
            _rpi.setup(_state.pins.dir_pin, GPIO.OUT, initial=GPIO.LOW)
            _BACKEND = "rpigpio"
            _state.initialized = True
            logger.info("Plunger ready (RPi.GPIO) EN=%s STEP=%s DIR=%s",
                        _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin)
        except Exception as e:
            _BACKEND = None
            _state.initialized = False
            logger.error("No usable GPIO backend (need pigpio or RPi.GPIO): %s", e, exc_info=True)
            raise RuntimeError("GPIO backend init failed")

def jog(steps: int, enable: bool = True, delay_us: int = 1200) -> Dict[str, object]:
    """Jog stepper; positive=forward, negative=reverse."""
    if not _state.initialized:
        raise RuntimeError("Plunger not initialized")
    if steps == 0:
        return {"ok": True, "steps": 0, "direction": "none", "backend": _BACKEND}
    if abs(steps) > _MAX_STEPS:
        raise ValueError(f"steps exceeds limit ({_MAX_STEPS})")
    delay_us = max(_MIN_DELAY_US, int(delay_us))

    direction_fwd = (steps > 0)
    if _state.pins.dir_invert:
        direction_fwd = not direction_fwd
    abs_steps = abs(steps)

    with _LOCK:
        prev = _state.enabled
        if enable:
            _set_enable_hw(True)
        _set_dir_hw(direction_fwd)
        if _BACKEND == "pigpio":
            _pulse_pigpio(abs_steps, delay_us)
        elif _BACKEND == "rpigpio":
            _pulse_rpigpio(abs_steps, delay_us)
        else:
            raise RuntimeError("No backend available")
        if enable and not prev:
            _set_enable_hw(False)

    return {"ok": True, "steps": steps, "direction": "forward" if steps > 0 else "reverse",
            "backend": _BACKEND, "delay_us": delay_us}

def shutdown() -> None:
    """Disable driver and release resources."""
    global _pi, _pigpio, _rpi, _BACKEND
    with _LOCK:
        try:
            if _state.initialized:
                _set_enable_hw(False)
        except Exception:
            logger.warning("Error disabling driver on shutdown", exc_info=True)
        try:
            if _BACKEND == "pigpio" and _pi is not None:
                _pi.stop()
        except Exception:
            logger.warning("Error stopping pigpio", exc_info=True)
        try:
            if _BACKEND == "rpigpio" and _rpi is not None:
                _rpi.cleanup()
        except Exception:
            logger.warning("GPIO cleanup failed", exc_info=True)
        _pi = None; _pigpio = None; _rpi = None
        _BACKEND = None
        _state.initialized = False
        _state.enabled = False
        logger.info("Plunger shutdown complete")

# --- helpers ---
def _inactive_level() -> int:
    return 0 if
