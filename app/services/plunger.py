# app/services/plunger.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

logger = logging.getLogger("plunger")

# =========================
# Backend selection/runtime
# =========================

_BACKEND: Optional[Literal["pigpio", "rpigpio"]] = None
_LOCK = threading.RLock()

# pigpio runtime
_pigpio = None  # module
_pi = None      # pigpio.pi()

# RPi.GPIO runtime
_rpi = None     # module

# =========================
# Configuration / State
# =========================

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

_MIN_DELAY_US = 200               # lower bound on half-period (us)
_MAX_STEPS = 500_000              # safety cap per jog call

# =========================
# Public API
# =========================

def init(cfg: Optional[Dict] = None) -> None:
    """
    Initialize the plunger driver.

    Optional cfg keys:
      enable_pin (int, BCM), step_pin (int), dir_pin (int),
      enable_active_low (bool), dir_invert (bool),
      backend: "pigpio" | "rpigpio" | None (auto-detect)
    """
    global _BACKEND, _pigpio, _pi, _rpi

    with _LOCK:
        if _state.initialized:
            logger.info("plunger.init(): already initialized (backend=%s)", _BACKEND)
            return

        # Apply configuration
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

        # Choose backend
        if forced in ("pigpio", "rpigpio"):
            choice = forced
        else:
            # Prefer pigpio if daemon available
            try:
                import pigpio as _tmp_pigpio  # type: ignore
                pi = _tmp_pigpio.pi()
                if not pi.connected:
                    raise RuntimeError("pigpio daemon not running")
                _pigpio = _tmp_pigpio
                _pi = pi
                choice = "pigpio"
            except Exception as e:
                logger.info("pigpio unavailable (%s); falling back to RPi.GPIO", e)
                choice = "rpigpio"

        if choice == "pigpio":
            # Configure pins
            _pi.set_mode(_state.pins.enable_pin, _pigpio.OUTPUT)
            _pi.set_mode(_state.pins.step_pin, _pigpio.OUTPUT)
            _pi.set_mode(_state.pins.dir_pin, _pigpio.OUTPUT)
            _set_enable_hw(False)
            _pi.write(_state.pins.step_pin, 0)
            _BACKEND = "pigpio"
            _state.initialized = True
            logger.info(
                "Plunger initialized (pigpio) EN=%s STEP=%s DIR=%s",
                _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin
            )
            return

        # Fallback: RPi.GPIO
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
            logger.info(
                "Plunger initialized (RPi.GPIO) EN=%s STEP=%s DIR=%s",
                _state.pins.enable_pin, _state.pins.step_pin, _state.pins.dir_pin
            )
        except Exception as e:
            _BACKEND = None
            _state.initialized = False
            logger.error("GPIO backend init failed (need pigpio or RPi.GPIO): %s", e, exc_info=True)
            raise RuntimeError("No usable GPIO backend found")

def jog(steps: int, enable: bool = True, delay_us: int = 1200) -> Dict[str, object]:
    """
    Jog the stepper by a signed number of steps.
      steps > 0 => forward
      steps < 0 => reverse

    Args:
      steps: number of steps (|steps| <= _MAX_STEPS)
      enable: assert ENABLE during motion, then restore previous state
      delay_us: half-period in microseconds (controls speed). Effective step rate ~ 1/(2*delay_us).
    """
    if not _state.initialized:
        raise RuntimeError("Plunger not initialized")

    if steps == 0:
        return {"ok": True, "steps": 0, "direction": "none", "backend": _BACKEND}

    if abs(steps) > _MAX_STEPS:
        raise ValueError(f"steps exceeds safety limit ({_MAX_STEPS})")

    delay_us = max(_MIN_DELAY_US, int(delay_us))

    # Determine direction and absolute steps
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
            raise RuntimeError("No GPIO backend available at runtime")

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
    """Disable the driver and release resources."""
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
            logger.warning("RPi.GPIO cleanup failed", exc_info=True)

        _pi = None
        _pigpio = None
        _rpi = None
        _BACKEND = None
        _state.initialized = False
        _state.enabled = False
        logger.info("Plunger shutdown complete")

# =========================
# Hardware helpers
# =========================

def _inactive_level() -> int:
    """GPIO level that disables the stepper driver."""
    return 0 if not _state.pins.enable_active_low else 1

def _active_level() -> int:
    """GPIO level that enables the stepper driver."""
    return 1 - _inactive_level()

def _set_enable_hw(value: bool) -> None:
    lvl = _active_level() if value else _inactive_level()
    if _BACKEND == "pigpio":
        _pi.write(_state.pins.enable_pin, lvl)
    elif _BACKEND == "rpigpio":
        _rpi.output(_state.pins.enable_pin, lvl)
    else:
        raise RuntimeError("No GPIO backend available")
    _state.enabled = value

def _set_dir_hw(forward: bool) -> None:
    lvl = 1 if forward else 0
    if _BACKEND == "pigpio":
        _pi.write(_state.pins.dir_pin, lvl)
    elif _BACKEND == "rpigpio":
        _rpi.output(_state.pins.dir_pin, lvl)
    else:
        raise RuntimeError("No GPIO backend available")

def _pulse_pigpio(steps: int, delay_us: int) -> None:
    """
    pigpio waveform for accurate pulse timing:
      (HIGH delay_us) + (LOW delay_us) repeated for each step.
    """
    step_gpio = _state.pins.step_pin
    pulses = []
    # pigpio.pulse(gpio_on, gpio_off, delay_us)
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
    """
    Software-timed pulsing via RPi.GPIO. Jitter is higher than pigpio,
    but acceptable at moderate speeds.
    """
    step_pin = _state.pins.step_pin
    on_t = delay_us / 1_000_000.0
    off_t = on_t
    for _ in range(steps):
        _rpi.output(step_pin, _rpi.HIGH)
        time.sleep(on_t)
        _rpi.output(step_pin, _rpi.LOW)
        time.sleep(off_t)
