# app/main.py
from __future__ import annotations

from typing import Tuple
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, BaseSettings, Field, validator

# Services
from app.services import camera, ocr, motion
import logging
logger = logging.getLogger("uvicorn.error")  # or your preferred logger

# app/main.py
from . import models
from .services import assign, camera, grid as grid_svc, motion, plunger
# if you have an OCR module:
from .services import ocr  # only if services/ocr.py exists


GRID: grid_svc.Grid | None = None

# -----------------------------------------------------------------------------
# App metadata / logging
# -----------------------------------------------------------------------------

APP_NAME = "Card Sorter"
APP_VERSION = "0.5.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("Card Sorter")

# -----------------------------------------------------------------------------
# Config models
# -----------------------------------------------------------------------------

class CameraConfig(BaseModel):
    # Accept whatever your camera service expects; typical fields shown
    device: Optional[str] = None
    backend: Optional[str] = None
    resolution: Optional[List[int]] = None   # [width, height]
    preview_fps: Optional[int] = None
    captures_dir: str = "captures"

class OCRConfig(BaseModel):
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 6
    oem: int = 1
    whitelist: Optional[str] = None

    # If your ocr.py exposes stricter selection/acceptance knobs, include them:
    min_accept_confidence: Optional[int] = 75
    top_priority_fraction: Optional[float] = 0.35
    top_bias_bonus: Optional[float] = 10.0
    min_line_chars: Optional[int] = 3

class MotionConfig(BaseModel):
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enabled: bool = False
    read_timeout_s: float = 2.0
    write_timeout_s: float = 2.0
    connect_timeout_s: float = 6.0
    reset_on_connect: bool = True
    startup_drain_s: float = 2.0
    ok_tokens: List[str] = ["ok"]
    error_tokens: List[str] = ["error", "Error:"]
    busy_tokens: List[str] = ["busy:", "wait"]

class AppConfig(BaseSettings):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)

    # (optional) explicitly capture other sections so you can read them later
    assignment: Optional[Dict[str, Any]] = None
    cors: Optional[Dict[str, Any]] = None
    grid: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    plunger: Optional[Dict[str, Any]] = None
    server: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"          # <- accept unknown top-level keys
        case_sensitive = False

# -----------------------------------------------------------------------------
# Load config.yaml
# -----------------------------------------------------------------------------

def load_config() -> AppConfig:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg_path.exists():
        log.warning("config.yaml not found at %s, using defaults", cfg_path)
        return AppConfig()  # all defaults
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return AppConfig(**raw)
    except Exception as e:
        raise RuntimeError(f"Invalid config.yaml: {e}") from e

CONFIG: AppConfig = load_config()

# Ensure captures directory exists
CAPTURES_DIR = Path(CONFIG.camera.captures_dir).resolve()
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# App + templates
# -----------------------------------------------------------------------------

app = FastAPI(title=APP_NAME, version=APP_VERSION)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# In-memory registries for IDs returned to the UI
CAPTURES: Dict[str, str] = {}     # capture_id -> file path
OCR_RUNS: Dict[str, Dict[str, Any]] = {}  # ocr_id -> result payload

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------
# --- put near the top with other imports ---
from typing import Optional, Tuple, List, Dict

def _auto_connect_motion(config_motion) -> None:
    """
    Politely try to connect when motion.enabled is true.
    Match order: by-id substring -> VID:PID -> configured port -> only port.
    Assumes motion.list_ports() returns a LIST[DICT].
    """
    try:
        if not bool(getattr(config_motion, "enabled", False)):
            return

        ports: List[Dict[str, str]] = motion.list_ports()  # LIST now
        if not ports:
            log.info("Motion auto-connect skipped: no serial ports present.")
            return

        by_id_sub = (getattr(config_motion, "by_id_contains", None) or "").strip().lower()
        want_vid = (getattr(config_motion, "vid", None) or "").strip().lower()
        want_pid = (getattr(config_motion, "pid", None) or "").strip().lower()
        conf_dev = (getattr(config_motion, "port", "") or "").strip()
        baud = int(getattr(config_motion, "baud", 250000))

        selected: Optional[Tuple[str, int]] = None

        # 1) by-id contains
        if by_id_sub:
            for p in ports:
                if by_id_sub in (p.get("by_id") or "").lower():
                    selected = ((p.get("by_id") or p.get("device")), baud)
                    log.info("Motion auto-connect: matched by by-id: %s", selected[0])
                    break

        # 2) VID/PID exact
        if selected is None and want_vid and want_pid:
            for p in ports:
                if p.get("vid", "").lower() == want_vid and p.get("pid", "").lower() == want_pid:
                    selected = (p["device"], baud)
                    log.info("Motion auto-connect: matched by VID:PID %s:%s -> %s", want_vid, want_pid, p["device"])
                    break

        # 3) Configured device
        if selected is None and conf_dev:
            for p in ports:
                if p.get("device") == conf_dev:
                    selected = (conf_dev, baud)
                    log.info("Motion auto-connect: using configured port %s", conf_dev)
                    break

        # 4) Only one present
        if selected is None and len(ports) == 1:
            only = ports[0].get("by_id") or ports[0].get("device")
            if only:
                selected = (only, baud)
                log.info("Motion auto-connect: single available port -> %s", only)

        if selected is None:
            devs = ", ".join([(p.get("by_id") or p.get("device", "?")) for p in ports])
            log.info("Motion auto-connect skipped: multiple ports (%s) or no match.", devs)
            return

        port, bd = selected
        motion.enable()
        motion.connect(port=port, baud=bd)

        # Optional probe (ignore failures)
        try:
            probe = motion.ping()
            first = (probe.get("lines") or probe.get("response") or [])[:1]
            log.info("Motion auto-connect OK on %s @ %d; probe: %s", port, bd, first)
        except Exception:
            log.info("Motion auto-connect connected on %s; ping failed (may still be fine).", port)

    except Exception as e:
        # Important: do NOT say list has no attribute get again
        log.info(f"Motion auto-connect skipped (helper error): {e}")
@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)

    # --- Camera ---
    try:
        camera.init(
            device=CONFIG.camera.device,
            backend=CONFIG.camera.backend,
            resolution=tuple(CONFIG.camera.resolution) if CONFIG.camera.resolution else None,
            captures_dir=str(CAPTURES_DIR),
            preview_fps=CONFIG.camera.preview_fps,
        )
    except TypeError:
        # fallback to dict-style init if your camera service expects a single config object
        camera.init({
            "device": CONFIG.camera.device,
            "backend": CONFIG.camera.backend,
            "resolution": CONFIG.camera.resolution,
            "captures_dir": str(CAPTURES_DIR),
            "preview_fps": CONFIG.camera.preview_fps,
        })
    except Exception as e:
        # Do not crash the app; surface clear error
        log.error("Camera init failed: %s", e, exc_info=True)

    # --- OCR ---
    try:
        ocr.init(cfg=CONFIG.ocr.dict())
    except Exception as e:
        log.error("OCR init failed: %s", e, exc_info=True)

    # --- Motion (init only; connection is polite and non-fatal) ---
    try:
        motion.init(CONFIG.motion.dict())
    except Exception as e:
        log.error("Motion init failed: %s", e, exc_info=True)

    # --- Polite auto-connect with short retry window (10s total) ---
    try:
        # Try immediately
        _auto_connect_motion(CONFIG.motion)

        # If not connected yet, retry once per second up to 10s to catch late USB enumeration
        for _ in range(10):
            if hasattr(motion, "is_connected") and motion.is_connected():
                break
            await asyncio.sleep(1.0)
            _auto_connect_motion(CONFIG.motion)
    except Exception as e:
        # Never fail startup on motion issues
        log.info("Motion auto-connect: retry loop skipped due to error: %s", e)

    log.info("Startup complete")


    # OCR
    ocr.init(cfg=CONFIG.ocr.dict())

    # Motion (init only; connection handled by helper)
    motion.init(CONFIG.motion.dict())

    # Attempt a *polite* auto-connect (won't warn if no ports)
    _auto_connect_motion(CONFIG.motion)

    log.info("Startup complete")

@app.on_event("startup")
async def on_startup():
    # ... your existing startup logic (load config, init camera/ocr, motion.init(...), etc.)
    try:
        _auto_motion_bringup(CONFIG)  # CONFIG is whatever var holds your loaded AppConfig
    except Exception as e:
        # Shouldn't happen since the helper already guards, but keep this belt-and-suspenders
        logger.warning("Startup: auto motion bring-up wrapper caught: %s", e)

@app.on_event("startup")
async def on_startup():
    # ... your existing startup code ...
    global GRID
    GRID = grid_svc.Grid(CONFIG.grid)
    # keep your existing auto-connect/enable bring-up


# --- Add this helper after you call motion.init(...) in main.py ---
def _auto_motion_bringup(cfg) -> None:
    """
    Try to connect and enable motion on startup.
    Uses cfg.motion.port/baud if present; otherwise picks the first available port.
    Never raises—logs warnings so the app still starts if the board is offline.
    """
    try:
        ports_info = motion.list_ports()                  # <-- this is already a LIST[DICT]
        if not isinstance(ports_info, list):
            # Defensive: normalize accidental shapes
            ports_info = list(ports_info or [])

        # Read configured values from the Pydantic model
        preferred = (cfg.motion.port or "").strip() if getattr(cfg, "motion", None) else ""
        baud = int(getattr(cfg.motion, "baud", 250000)) if getattr(cfg, "motion", None) else 250000

        # Flatten available device paths
        avail = []
        for p in ports_info:
            if not isinstance(p, dict):
                continue
            dev = (p.get("by_id") or p.get("device") or "").strip()
            if dev:
                avail.append(dev)

        if not avail:
            logger.info("Motion auto-connect: no serial ports found.")
            return

        use_port = preferred if preferred in avail else avail[0]
        if preferred and preferred not in avail:
            logger.info("Motion auto-connect: preferred %s not found; using %s", preferred, use_port)

        motion.init(cfg.motion.dict())      # idempotent; safe if already called
        motion.enable()
        motion.connect(port=use_port, baud=baud)

        # Optional probe; non-fatal if firmware doesn’t respond
        try:
            motion.ping()
        except Exception as e:
            logger.info("Motion auto-connect: connected on %s @ %d; ping warn: %s", use_port, baud, e)

        logger.info("Motion auto-connect: connected on %s @ %d and enabled.", use_port, baud)

    except Exception as e:
        logger.warning("Motion auto-connect failed: %s", e)

@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)

    # Camera
    try:
        camera.init(
            device=CONFIG.camera.device,
            backend=CONFIG.camera.backend,
            resolution=tuple(CONFIG.camera.resolution) if CONFIG.camera.resolution else None,
            captures_dir=str(CAPTURES_DIR),
            preview_fps=CONFIG.camera.preview_fps,
        )
    except TypeError:
        camera.init({
            "device": CONFIG.camera.device,
            "backend": CONFIG.camera.backend,
            "resolution": CONFIG.camera.resolution,
            "captures_dir": str(CAPTURES_DIR),
            "preview_fps": CONFIG.camera.preview_fps,
        })
    except Exception as e:
        log.error("Camera init failed: %s", e, exc_info=True)

    # OCR
    try:
        ocr.init(cfg=CONFIG.ocr.dict())
    except Exception as e:
        log.error("OCR init failed: %s", e, exc_info=True)

    # Motion (init; connection handled by helper)
    try:
        motion.init(CONFIG.motion.dict())
    except Exception as e:
        log.error("Motion init failed: %s", e, exc_info=True)

    # Try immediate auto-connect; if not present yet, retry briefly to catch late enumeration
    try:
        _auto_motion_bringup(CONFIG)
        for _ in range(10):
            if hasattr(motion, "is_connected") and motion.is_connected():
                break
            await asyncio.sleep(1.0)
            _auto_motion_bringup(CONFIG)
    except Exception as e:
        log.info("Motion auto-connect retry loop skipped due to error: %s", e)

    log.info("Startup complete")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    try:
        motion.disconnect()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Templated UI
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "version": APP_VERSION,
        },
    )

# -----------------------------------------------------------------------------
# Camera endpoints
# -----------------------------------------------------------------------------

def _camera_stream_gen():
    """
    Use the camera service's MJPEG generator. Supports either:
      - camera.mjpeg_stream()
      - camera.stream()
    """
    if hasattr(camera, "mjpeg_stream"):
        return camera.mjpeg_stream()
    if hasattr(camera, "stream"):
        return camera.stream()
    raise RuntimeError("Camera service does not provide an MJPEG stream generator")

def _fallback_stream():
    """
    Fallback MJPEG generator if the camera service doesn't expose an MJPEG stream generator.
    It tries:
      - camera.get_jpeg_frame() -> bytes
      - camera.get_frame() -> np.ndarray (BGR), which we JPEG-encode
    """
    import cv2
    boundary = b"--frame"
    while True:
        try:
            # Preferred: service returns JPEG bytes directly
            if hasattr(camera, "get_jpeg_frame"):
                jpg = camera.get_jpeg_frame()  # must be bytes
                if not isinstance(jpg, (bytes, bytearray)):
                    raise RuntimeError("camera.get_jpeg_frame must return bytes")
            else:
                # Try raw frame then encode
                if not hasattr(camera, "get_frame"):
                    # Nothing available; back off
                    time.sleep(0.2)
                    continue
                frame = camera.get_frame()  # np.ndarray (BGR)
                if frame is None:
                    time.sleep(0.02)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    time.sleep(0.02)
                    continue
                jpg = bytes(buf)

            # Emit a compliant MJPEG part
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                b"\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.05)  # ~20 fps
        except GeneratorExit:
            break
        except Exception:
            # transient failure; keep streaming
            time.sleep(0.2)

def _mjpeg_source():
    # Prefer a native generator if the service provides one
    if hasattr(camera, "mjpeg_stream"):
        gen = camera.mjpeg_stream()
        if gen is not None:
            return gen
    if hasattr(camera, "stream"):
        gen = camera.stream()
        if gen is not None:
            return gen
    # else build one
    return _fallback_stream()

@app.get("/camera/stream")
def camera_stream():
    try:
        gen = _mjpeg_source()
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Connection": "close",
        }
        return StreamingResponse(
            gen,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"camera stream error: {e}")

@app.get("/camera/capture")
def camera_capture():
    """
    Capture a still image; return a capture_id and the absolute path.
    """
    try:
        path = camera.capture()  # your camera service should save into CAPTURES_DIR
        if not path:
            raise RuntimeError("Camera capture returned no path")
        abs_path = str(Path(path).resolve())
        capture_id = uuid.uuid4().hex[:8]
        CAPTURES[capture_id] = abs_path
        return {"capture_id": capture_id, "path": abs_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# OCR endpoints
# -----------------------------------------------------------------------------

@app.post("/ocr/{capture_id}")
def ocr_on_capture(capture_id: str):
    """
    Run OCR against a previously captured image by capture_id.
    Returns the ocr result and an ocr_id for auditing.
    """
    path = CAPTURES.get(capture_id)
    if not path:
        raise HTTPException(404, f"Unknown capture_id: {capture_id}")

    try:
        result = ocr.run(path)
        ocr_id = uuid.uuid4().hex[:8]
        payload = {
            "ocr_id": ocr_id,
            "capture_id": capture_id,
            **result,
        }
        OCR_RUNS[ocr_id] = payload
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Simple pipeline (Capture → OCR). Returns IDs + OCR text/confidence.
# -----------------------------------------------------------------------------

@app.post("/sort")
def sort_one():
    try:
        # Capture
        path = camera.capture()
        if not path:
            raise RuntimeError("Camera capture returned no path")
        abs_path = str(Path(path).resolve())
        capture_id = uuid.uuid4().hex[:8]
        CAPTURES[capture_id] = abs_path

        # OCR (full image, with your prioritization inside ocr.py if configured)
        ocr_result = ocr.run(abs_path)
        ocr_id = uuid.uuid4().hex[:8]
        OCR_RUNS[ocr_id] = {"ocr_id": ocr_id, "capture_id": capture_id, **ocr_result}

        # If you want to gate progression on confidence / acceptance, check here:
        accepted = bool(ocr_result.get("accepted", True))
        if not accepted:
            # Do not plan or send motion if low quality; surface a clear message
            return {
                "capture_id": capture_id,
                "ocr_id": ocr_id,
                "assignment_id": None,
                "move_id": None,
                "text": ocr_result.get("text", ""),
                "confidence": ocr_result.get("confidence", 0),
                "message": "OCR confidence below threshold; not proceeding to motion planning.",
            }

        # For now we stop after OCR (no assignment/grid planning in this minimal main).
        return {
            "capture_id": capture_id,
            "ocr_id": ocr_id,
            "assignment_id": None,
            "move_id": None,
            "text": ocr_result.get("text", ""),
            "confidence": ocr_result.get("confidence", 0),
        }

    except Exception as e:
        log.error("Sort pipeline failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Motion endpoints
# -----------------------------------------------------------------------------

class GCodeBody(BaseModel):
    gcode: List[str]

@app.get("/motion/ports")
def motion_ports():
    try:
        return {"ports": motion.list_ports()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/connect")
def motion_connect(port: Optional[str] = None, baud: Optional[int] = None):
    try:
        return motion.connect(port=port, baud=baud)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/enable")
def motion_enable():
    try:
        return motion.enable()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/disable")
def motion_disable():
    try:
        return motion.disable()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/motion/ping")
def motion_ping():
    try:
        return motion.ping()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/home")
def motion_home(axes: str = "XY"):
    try:
        return motion.home(axes)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/test-square")
def motion_test_square(size_mm: float = 20.0, feed_xy: int = 1200):
    try:
        return motion.test_square(size_mm=size_mm, feed_xy=feed_xy)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/send")
def motion_send(body: GCodeBody):
    try:
        return motion.send(body.gcode)
    except Exception as e:
        raise HTTPException(500, str(e))
# add near other motion endpoints
@app.post("/motion/jog")
def motion_jog(axis: str, delta_mm: float, feed_xy: int = 1200):
    try:
        return motion.jog(axis=axis, delta_mm=delta_mm, feed_xy=feed_xy)
    except Exception as e:
        raise HTTPException(500, str(e))
from fastapi import HTTPException

@app.get("/grid/zones")
def grid_list_zones():
    try:
        return {"zones": GRID.list_zones()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/zone")
def grid_move_zone(key: str, safe_lift: bool = True):
    try:
        z = GRID.get_zone(key)
        return motion.goto_xy(z.x_mm, z.y_mm, safe_lift=safe_lift)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/rc")
def grid_move_rc(row: int, col: int, safe_lift: bool = True):
    try:
        x, y = GRID.slot_rc_to_xy(row, col)
        return motion.goto_xy(x, y, safe_lift=safe_lift)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/slotid")
def grid_move_slotid(slot_id: int, safe_lift: bool = True):
    try:
        r, c = GRID.slotid_to_rc(slot_id)
        x, y = GRID.slot_rc_to_xy(r, c)
        return motion.goto_xy(x, y, safe_lift=safe_lift)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/letter")
def grid_move_letter(letter: str, safe_lift: bool = True):
    try:
        z = GRID.letter_to_zone(letter)
        if not z:
            raise HTTPException(404, f"No zone mapped for letter '{letter}'")
        return motion.goto_xy(z.x_mm, z.y_mm, safe_lift=safe_lift)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

