# app/main.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    HTTPException,
    Path as FPath,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

# Local modules
from app import models
from app.services import camera, ocr, assign as assign_srv, motion, plunger

APP_NAME = "Card Sorter"
APP_VERSION = "0.4.0"

# ======================================================================
# Config models
# ======================================================================

class OCRConfig(BaseModel):
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 6
    whitelist: Optional[str] = None
    enable_orientation: bool = True
    enable_deskew: bool = True
    gamma: float = 1.0
    denoise: bool = True
    unsharp: bool = True
    adaptive_thresh: bool = True
    invert_if_needed: bool = True
    otsu_fallback: bool = True


class CameraConfig(BaseModel):
    device: Optional[str] = None          # e.g., "/dev/video0"; None => prefer Picamera2
    resolution: Tuple[int, int] = (1280, 720)
    preview_fps: int = 5


class MotionConfig(BaseModel):
    port: str = "/dev/ttyACM0"
    baud: int = 250000
    enable: bool = False
    homing_sequence: List[str] = Field(default_factory=lambda: ["G28"])
    feedrates: Dict[str, int] = Field(default_factory=lambda: {"xy": 3000, "z": 600})


class GridConfig(BaseModel):
    origin_mm: Tuple[float, float] = (0.0, 0.0)
    pitch_mm: Tuple[float, float] = (70.0, 95.0)
    rows: int = 5
    cols: int = 10
    z: Dict[str, float] = Field(default_factory=lambda: {"pick": -2.0, "travel": 15.0, "safe": 20.0})


class AssignmentConfig(BaseModel):
    mode: str = "first_letter"            # or "csv_lookup"
    slots_csv: Optional[str] = None
    default_slot: Optional[int] = None
    first_letter_groups: Optional[Dict[str, int]] = None  # allow null


class PlungerConfig(BaseModel):
    enable_pin: int = 18
    step_pin: int = 23
    dir_pin: int = 24
    enable_active_low: bool = True
    dir_invert: bool = False
    backend: Optional[str] = None         # "pigpio" | "rpigpio" | None (auto)


class AppConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    ocr: OCRConfig = OCRConfig()
    assignment: AssignmentConfig = AssignmentConfig()
    grid: GridConfig = GridConfig()
    motion: MotionConfig = MotionConfig()
    plunger: PlungerConfig = PlungerConfig()
    server: Dict[str, Any] = Field(default_factory=lambda: {"host": "0.0.0.0", "port": 8000})
    logging: Dict[str, Any] = Field(default_factory=lambda: {"level": "INFO"})
    cors: Dict[str, Any] = Field(default_factory=lambda: {"allow_origins": ["*"]})


def load_config() -> AppConfig:
    cfg_path = Path("config.yaml")
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            raw = yaml.safe_load(f) or {}
    try:
        return AppConfig(**raw)
    except ValidationError as e:
        raise RuntimeError(f"Invalid config.yaml: {e}")


CONFIG: AppConfig = load_config()

# ======================================================================
# Logging
# ======================================================================

LOG_LEVEL = getattr(logging, str(CONFIG.logging.get("level", "INFO")).upper(), logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(APP_NAME)

# ======================================================================
# FastAPI app bootstrap
# ======================================================================

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Static & templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.cors.get("allow_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# Pydantic response models
# ======================================================================

class HealthResponse(BaseModel):
    status: str
    version: str

class StatusResponse(BaseModel):
    camera: bool
    ocr: bool
    motion_enabled: bool
    motion_port: Optional[str] = None
    db_ok: bool

class CaptureResponse(BaseModel):
    capture_id: int
    path: str

class OCRResponse(BaseModel):
    ocr_id: int
    text: str
    confidence: int
    boxes: Dict[str, Any]

class AssignmentResponse(BaseModel):
    assignment_id: int
    slot_id: Optional[int]
    rule_used: str

class MovePlanResponse(BaseModel):
    move_id: Optional[int] = None
    gcode: List[str]

class ToggleResponse(BaseModel):
    ok: bool
    enabled: Optional[bool] = None
    error: Optional[str] = None

class ListCapturesResponse(BaseModel):
    total: int
    items: List[Dict[str, Any]]

# ======================================================================
# Error handlers
# ======================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ======================================================================
# Lifecycle
# ======================================================================

@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting %s v%s", APP_NAME, APP_VERSION)

    # DB & schema
    models.init_db()

    # Services
    camera.init(
        device=CONFIG.camera.device,
        resolution=CONFIG.camera.resolution,
        preview_fps=CONFIG.camera.preview_fps,
    )
    ocr.init(cfg=CONFIG.ocr.dict())
    assign_srv.init(cfg={
        "mode": CONFIG.assignment.mode,
        "default_slot": CONFIG.assignment.default_slot,
        "slots_csv": CONFIG.assignment.slots_csv,
        "first_letter_groups": CONFIG.assignment.first_letter_groups or {},
    })
    motion.init(cfg={
        "port": CONFIG.motion.port,
        "baud": CONFIG.motion.baud,
        "enabled": CONFIG.motion.enable,
        "homing_sequence": CONFIG.motion.homing_sequence,
        "feedrates": CONFIG.motion.feedrates,
        "grid": CONFIG.grid.dict(),
    })
    try:
        plunger.init(cfg=CONFIG.plunger.dict())
    except Exception as e:
        logger.warning("Plunger init soft-failed: %s", e)

    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down %s", APP_NAME)
    try:
        camera.shutdown()
    except Exception:
        logger.warning("Camera shutdown issue", exc_info=True)
    try:
        motion.shutdown()
    except Exception:
        logger.warning("Motion shutdown issue", exc_info=True)
    try:
        plunger.shutdown()
    except Exception:
        logger.warning("Plunger shutdown issue", exc_info=True)

# Graceful termination for uvicorn/gunicorn
def _install_signal_handlers():
    loop = asyncio.get_event_loop()
    def handle_sigterm():
        logger.info("SIGTERM received, stopping loop")
        loop.stop()
    try:
        loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
    except NotImplementedError:
        pass
_install_signal_handlers()

# ======================================================================
# WebSocket for live status
# ======================================================================

class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = ConnectionManager()

@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            payload = {
                "camera": camera.status(),
                "ocr": ocr.status(),
                "motion_enabled": motion.enabled(),
                "db_ok": models.health_check(),
            }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)
        logger.exception("WebSocket error")

# ======================================================================
# Routes
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": APP_NAME, "version": APP_VERSION},
    )

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)

@app.get("/version")
async def version():
    return {"name": APP_NAME, "version": APP_VERSION}

@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    return StatusResponse(
        camera=camera.status(),
        ocr=ocr.status(),
        motion_enabled=motion.enabled(),
        motion_port=motion.port_or_none(),
        db_ok=models.health_check(),
    )

# ---- Captures / OCR / Assignment ----

@app.get("/captures", response_model=ListCapturesResponse)
async def list_captures(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    total, rows = models.list_captures(page=page, page_size=page_size)
    return ListCapturesResponse(total=total, items=rows)

@app.get("/captures/{capture_id}")
async def get_capture(capture_id: int = FPath(..., ge=1)):
    row = models.get_capture(capture_id)
    if not row:
        raise HTTPException(status_code=404, detail="Capture not found")
    return row

@app.post("/capture", response_model=CaptureResponse)
async def do_capture(background_tasks: BackgroundTasks) -> CaptureResponse:
    try:
        path = camera.capture()
        capture_id = models.record_capture(path)
        background_tasks.add_task(models.generate_thumbnail, path)
        await ws_manager.broadcast({"event": "capture", "capture_id": capture_id})
        return CaptureResponse(capture_id=capture_id, path=path)
    except Exception as e:
        logger.exception("Camera capture failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/{capture_id}", response_model=OCRResponse)
async def do_ocr(capture_id: int = FPath(..., ge=1)) -> OCRResponse:
    cap = models.get_capture(capture_id)
    if not cap:
        raise HTTPException(status_code=404, detail="Capture not found")
    try:
        result = ocr.run(cap["path"])
        ocr_id = models.record_ocr(capture_id, result)
        await ws_manager.broadcast({"event": "ocr", "ocr_id": ocr_id})
        return OCRResponse(ocr_id=ocr_id, **result)
    except Exception as e:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assign/{ocr_id}", response_model=AssignmentResponse)
async def assign_slot(ocr_id: int = FPath(..., ge=1)) -> AssignmentResponse:
    row = models.get_ocr(ocr_id)
    if not row:
        raise HTTPException(status_code=404, detail="OCR result not found")
    decision = assign_srv.decide(row["text"])
    assignment_id = models.record_assignment(ocr_id, decision)
    await ws_manager.broadcast({"event": "assign", "assignment_id": assignment_id})
    return AssignmentResponse(assignment_id=assignment_id, **decision)

# ---- Motion: planning / queue / execute ----

@app.post("/motion/plan/{assignment_id}", response_model=MovePlanResponse)
async def motion_plan(assignment_id: int = FPath(..., ge=1)) -> MovePlanResponse:
    try:
        gcode = motion.plan_for(assignment_id)
        return MovePlanResponse(move_id=None, gcode=gcode)
    except Exception as e:
        logger.exception("Plan failed")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/motion/queue/{assignment_id}", response_model=MovePlanResponse)
async def motion_queue(assignment_id: int = FPath(..., ge=1)) -> MovePlanResponse:
    try:
        gcode = motion.plan_for(assignment_id)
        move_id = models.enqueue_move(assignment_id, gcode)
        await ws_manager.broadcast({"event": "queue", "move_id": move_id})
        return MovePlanResponse(move_id=move_id, gcode=gcode)
    except Exception as e:
        logger.exception("Queue failed")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/motion/execute/{move_id}")
async def motion_execute(move_id: int = FPath(..., ge=1)):
    try:
        res = motion.execute(move_id)
        if not res.get("ok"):
            raise HTTPException(status_code=400, detail=res.get("error", "Execution failed"))
        await ws_manager.broadcast({"event": "execute", "move_id": move_id})
        return res
    except Exception as e:
        logger.exception("Execute failed")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/motion/enable", response_model=ToggleResponse)
async def motion_enable() -> ToggleResponse:
    try:
        enabled = motion.set_enabled(True)
        await ws_manager.broadcast({"event": "motion_enabled", "value": True})
        return ToggleResponse(ok=True, enabled=enabled)
    except Exception as e:
        logger.exception("Enable motion failed")
        return ToggleResponse(ok=False, error=str(e))

@app.post("/motion/disable", response_model=ToggleResponse)
async def motion_disable() -> ToggleResponse:
    try:
        enabled = motion.set_enabled(False)
        await ws_manager.broadcast({"event": "motion_enabled", "value": False})
        return ToggleResponse(ok=True, enabled=enabled)
    except Exception as e:
        logger.exception("Disable motion failed")
        return ToggleResponse(ok=False, error=str(e))

# ---- Motion: new diagnostics / raw send ----

@app.get("/motion/ports")
async def motion_ports():
    return {"ports": motion.list_ports()}

@app.get("/motion/ping")
async def motion_ping():
    res = motion.ping()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "ping failed"))
    return res

@app.post("/motion/home")
async def motion_home():
    res = motion.home_now()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "home failed"))
    await ws_manager.broadcast({"event": "motion_home"})
    return res

@app.post("/motion/send")
async def motion_send(body: dict = Body(...)):
    """
    Body example: {"gcode": ["M115", "G0 X10 Y10 F3000", "M400"]}
    """
    gcode = body.get("gcode") or body.get("lines")
    if not isinstance(gcode, list):
        raise HTTPException(status_code=422, detail="Body must include list 'gcode'")
    res = motion.send_raw([str(x) for x in gcode])
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "send failed"))
    await ws_manager.broadcast({"event": "motion_send", "count": res.get("sent", 0)})
    return res

@app.post("/motion/test-square")
async def motion_test_square(
    size_mm: float = Query(20.0, gt=0),
    feed_xy: int = Query(1200, gt=0),
):
    """
    Draw a small square in XY at current Z (no Z moves); controller must be homed or in a safe state.
    """
    if not motion.enabled():
        raise HTTPException(status_code=400, detail="Motion is disabled")
    lines = [
        "G90",
        f"G1 F{feed_xy}",
        "M400",
        "G4 P50",
        "G1 X0 Y0",
        f"G1 X{size_mm} Y0",
        f"G1 X{size_mm} Y{size_mm}",
        f"G1 X0 Y{size_mm}",
        "G1 X0 Y0",
        "M400",
    ]
    res = motion.send_raw(lines)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "send failed"))
    await ws_manager.broadcast({"event": "motion_test_square", "size": size_mm})
    return res

# ---- Camera helpers ----

@app.get("/camera/snapshot")
async def camera_snapshot():
    try:
        jpeg = camera.grab_jpeg()
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        logger.exception("Snapshot failed")
        return PlainTextResponse(str(e), status_code=500)

@app.get("/camera/stream")
async def camera_stream():
    """
    MJPEG stream for <img src="/camera/stream">.
    """
    async def gen():
        boundary = b"--frame\r\n"
        while True:
            try:
                jpeg = camera.grab_jpeg()
                yield boundary
                yield b"Content-Type: image/jpeg\r\n"
                yield b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                yield jpeg + b"\r\n"
            except Exception as e:
                logger.debug("Stream frame error: %s", e)
                await asyncio.sleep(0.2)
            # ~preview_fps
            await asyncio.sleep(max(0.01, 1.0 / max(CONFIG.camera.preview_fps, 1)))
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/capture", response_model=CaptureResponse)
async def camera_capture_alias(background_tasks: BackgroundTasks) -> CaptureResponse:
    try:
        path = camera.capture()
        capture_id = models.record_capture(path)
        background_tasks.add_task(models.generate_thumbnail, path)
        await ws_manager.broadcast({"event": "capture", "capture_id": capture_id})
        return CaptureResponse(capture_id=capture_id, path=path)
    except Exception as e:
        logger.exception("Camera capture failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---- One-button pipeline: capture → OCR → assign → plan (queue) ----

@app.post("/sort")
async def sort_one():
    try:
        # Capture
        path = camera.capture()
        capture_id = models.record_capture(path)

        # OCR
        ocr_result = ocr.run(path)
        ocr_id = models.record_ocr(capture_id, ocr_result)

        # Assign
        decision = assign_srv.decide(ocr_result["text"])
        assignment_id = models.record_assignment(ocr_id, decision)

        # Plan + queue (do not execute by default)
        gcode = motion.plan_for(assignment_id)
        move_id = models.enqueue_move(assignment_id, gcode)

        payload = {
            "capture_id": capture_id,
            "ocr_id": ocr_id,
            "assignment_id": assignment_id,
            "move_id": move_id,
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
            "slot_id": decision.get("slot_id"),
            "rule_used": decision.get("rule_used"),
            "gcode": gcode,
        }
        await ws_manager.broadcast({"event": "sort", **payload})
        return payload
    except Exception as e:
        logger.exception("Sort pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
# Dev entry point
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=CONFIG.server.get("host", "0.0.0.0"),
        port=int(CONFIG.server.get("port", 8000)),
        reload=True,
    )
