# app/main.py
from __future__ import annotations

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

@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)

    # Initialize services
    try:
        camera.init(
            device=CONFIG.camera.device,
            backend=CONFIG.camera.backend,
            resolution=tuple(CONFIG.camera.resolution) if CONFIG.camera.resolution else None,
            captures_dir=str(CAPTURES_DIR),
            preview_fps=CONFIG.camera.preview_fps,
        )
    except TypeError:
        # Fallback if your camera.init takes a single dict
        camera.init({
            "device": CONFIG.camera.device,
            "backend": CONFIG.camera.backend,
            "resolution": CONFIG.camera.resolution,
            "captures_dir": str(CAPTURES_DIR),
            "preview_fps": CONFIG.camera.preview_fps,
        })

    ocr.init(cfg=CONFIG.ocr.dict())

    motion.init(CONFIG.motion.dict())
    # Do not auto-enable/connect; leave that to explicit UI actions

    log.info("Startup complete")
@app.on_event("startup")
async def on_startup():
    # ... your existing startup logic (load config, init camera/ocr, motion.init(...), etc.)
    try:
        _auto_motion_bringup(CONFIG)  # CONFIG is whatever var holds your loaded AppConfig
    except Exception as e:
        # Shouldn't happen since the helper already guards, but keep this belt-and-suspenders
        logger.warning("Startup: auto motion bring-up wrapper caught: %s", e)

# --- Add this helper after you call motion.init(...) in main.py ---
def _auto_motion_bringup(cfg):
    """
    Try to connect and enable motion on startup.
    Uses cfg.motion.port/baud if present; otherwise picks the first available port.
    Never raises (logs warnings instead) so the app still starts if the board is offline.
    """
    try:
        ports_info = motion.list_ports().get("ports", [])
        # Candidate port from config (if provided)
        preferred = (getattr(cfg, "motion", None) or {}).get("port", None) if isinstance(cfg.motion, dict) \
                    else getattr(cfg.motion, "port", None)
        baud = (getattr(cfg, "motion", None) or {}).get("baud", 250000) if isinstance(cfg.motion, dict) \
               else getattr(cfg.motion, "baud", 250000)

        # Normalize available devices
        avail = [p.get("device") for p in ports_info if p.get("device")]
        if not avail:
            logger.warning("Motion auto-connect: no serial ports found.")
            return

        # Pick a port: use preferred if it is in the list; else first available
        use_port = preferred if preferred in avail else avail[0]
        if preferred and preferred not in avail:
            logger.warning("Motion auto-connect: preferred port %s not found; using %s", preferred, use_port)

        motion.connect(port=use_port, baud=baud)
        motion.enable()
        try:
            motion.ping()
        except Exception as e:
            # Not fatal; some firmware may not respond to M115 as expected
            logger.warning("Motion auto-connect: ping warning: %s", e)

        logger.info("Motion auto-connect: connected on %s @ %s and enabled.", use_port, baud)

    except Exception as e:
        logger.warning("Motion auto-connect failed: %s", e)
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

@app.get("/camera/stream")
def camera_stream():
    try:
        gen = _camera_stream_gen()
        return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

