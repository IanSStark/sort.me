# app/main.py
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, BaseSettings, Field

# Local services
from .services import camera, ocr, motion, plunger, assign, grid as grid_svc  # services live under app/services
from . import models  # optional DB helpers

# -----------------------------------------------------------------------------
# App metadata / logging
# -----------------------------------------------------------------------------
APP_NAME = "Card Sorter"
APP_VERSION = "0.6.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(APP_NAME)

# -----------------------------------------------------------------------------
# Config models
# -----------------------------------------------------------------------------
class CameraConfig(BaseModel):
    device: Optional[str] = None
    backend: Optional[str] = None
    resolution: Optional[List[int]] = None   # [w,h]
    preview_fps: Optional[int] = 10
    captures_dir: str = "captures"

class OCRConfig(BaseModel):
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 6
    oem: int = 1
    whitelist: Optional[str] = None
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
    # Optional selectors and convenience params (passed through if present in YAML)
    by_id_contains: Optional[str] = None
    vid: Optional[str] = None
    pid: Optional[str] = None
    absolute_mode: Optional[bool] = None
    safe_z_mm: Optional[float] = None
    travel_feed_xy: Optional[int] = None
    travel_feed_z: Optional[int] = None
    home_on_start: Optional[bool] = None
    home_axes: Optional[str] = None
    standby_pos: Optional[Dict[str, float]] = None
    test_square: Optional[Dict[str, Any]] = None

class AppConfig(BaseSettings):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)

    # pass-through sections so we can forward them to services
    grid: Optional[Dict[str, Any]] = None
    plunger: Optional[Dict[str, Any]] = None
    assignment: Optional[Dict[str, Any]] = None
    server: Optional[Dict[str, Any]] = None
    cors: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"
        case_sensitive = False

# -----------------------------------------------------------------------------
# Load config.yaml
# -----------------------------------------------------------------------------
def load_config() -> AppConfig:
    # config.yaml one level up from this file (project root /app/config.yaml)
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg_path.exists():
        log.warning("config.yaml not found at %s; using defaults", cfg_path)
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return AppConfig(**raw)
    except Exception as e:
        raise RuntimeError(f"Invalid config.yaml: {e}") from e

CONFIG: AppConfig = load_config()

# -----------------------------------------------------------------------------
# Globals & framework
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME, version=APP_VERSION)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# Runtime helpers
GRID: Optional[grid_svc.Grid] = None
CAPTURES: Dict[str, str] = {}              # capture_id -> absolute image path
OCR_RUNS: Dict[str, Dict[str, Any]] = {}   # ocr_id -> result dict

# Derived motion convenience with sane defaults if YAML omitted these
_DEF_SAFE_Z = float(CONFIG.motion.safe_z_mm if CONFIG.motion.safe_z_mm is not None else 5.0)
_DEF_FEED_XY = int(CONFIG.motion.travel_feed_xy if CONFIG.motion.travel_feed_xy is not None else 1800)
_DEF_FEED_Z  = int(CONFIG.motion.travel_feed_z  if CONFIG.motion.travel_feed_z  is not None else 600)

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------
def _auto_connect_motion() -> None:
    """
    Attempt a polite auto-connect based on by-id substring, VID/PID, or configured port.
    Never raises.
    """
    try:
        if not CONFIG.motion.enabled:
            return

        ports = motion.list_ports()  # returns LIST[DICT] (device metadata)  :contentReference[oaicite:7]{index=7}
        if not ports:
            log.info("Motion auto-connect: no serial ports present.")
            return

        by_id_sub = (CONFIG.motion.by_id_contains or "").strip().lower()
        want_vid = (CONFIG.motion.vid or "").strip().lower()
        want_pid = (CONFIG.motion.pid or "").strip().lower()
        conf_dev = (CONFIG.motion.port or "").strip()
        baud = int(CONFIG.motion.baud or 250000)

        def pick() -> Optional[Tuple[str, int]]:
            # 1) by-id contains
            if by_id_sub:
                for p in ports:
                    if by_id_sub in (p.get("by_id") or "").lower():
                        return ((p.get("by_id") or p.get("device")), baud)
            # 2) VID/PID exact
            if want_vid and want_pid:
                for p in ports:
                    if p.get("vid", "").lower() == want_vid and p.get("pid", "").lower() == want_pid:
                        return (p["device"], baud)
            # 3) configured device if present in list
            if conf_dev:
                for p in ports:
                    if p.get("device") == conf_dev:
                        return (conf_dev, baud)
            # 4) single available
            if len(ports) == 1:
                only = ports[0].get("by_id") or ports[0].get("device")
                if only:
                    return (only, baud)
            # 5) fallback to first
            if ports:
                return (ports[0].get("device"), baud)
            return None

        chosen = pick()
        if not chosen:
            log.info("Motion auto-connect: no suitable port selection.")
            return

        port, bd = chosen
        motion.enable()
        motion.connect(port=port, baud=bd)  # open serial & drain banner  :contentReference[oaicite:8]{index=8}
        try:
            motion.ping()  # optional probe; ignore failures
        except Exception:
            pass
        log.info("Motion auto-connect OK: %s @ %d", port, bd)
    except Exception as e:
        log.info("Motion auto-connect skipped: %s", e)

@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)

    # Initialize DB (optional, safe if unused)
    try:
        models.init_db()  # creates tables if needed  :contentReference[oaicite:9]{index=9}
    except Exception as e:
        log.info("DB init skipped: %s", e)

    # Build GRID from YAML
    try:
        global GRID
        GRID = grid_svc.Grid(CONFIG.grid or {})  # grid math & zones  :contentReference[oaicite:10]{index=10}
    except Exception as e:
        raise RuntimeError(f"Grid initialization failed: {e}")

    # Camera
    try:
        camera.init(
            device=CONFIG.camera.device,
            backend=CONFIG.camera.backend,
            resolution=tuple(CONFIG.camera.resolution) if CONFIG.camera.resolution else (1280, 720),
            captures_dir=CONFIG.camera.captures_dir,
            preview_fps=CONFIG.camera.preview_fps or 10,
        )  # camera service  :contentReference[oaicite:11]{index=11}
        log.info("Camera initialized.")
    except Exception as e:
        log.error("Camera init failed: %s", e, exc_info=True)

    # OCR
    try:
        ocr.init(cfg=CONFIG.ocr.dict())  # OCR engine init  :contentReference[oaicite:12]{index=12}
        log.info("OCR initialized.")
    except Exception as e:
        log.error("OCR init failed: %s", e, exc_info=True)

    # Motion (configure; do not fail app if board is absent)
    try:
        motion.init(CONFIG.motion.dict())  # serial & firmware settings  :contentReference[oaicite:13]{index=13}
        _auto_connect_motion()
    except Exception as e:
        log.info("Motion init skipped: %s", e)

    # Plunger (GPIO/stepper for vacuum/testing)
    try:
        plunger.init(CONFIG.plunger or {})  # GPIO backend selection  :contentReference[oaicite:14]{index=14}
    except Exception as e:
        log.info("Plunger init skipped: %s", e)

    log.info("Startup complete.")

@app.on_event("shutdown")
async def on_shutdown() -> None:
    try:
        camera.shutdown()
    except Exception:
        pass
    try:
        plunger.shutdown()
    except Exception:
        pass
    try:
        motion.close()
    except Exception:
        pass
    log.info("Shutdown complete.")

# -----------------------------------------------------------------------------
# Small travel planner (absolute XY with optional Z safe-lift) via G-code
# -----------------------------------------------------------------------------
def _plan_xy_move_gcode(x_mm: float, y_mm: float, safe_lift: bool = True) -> List[str]:
    """
    Compose a safe absolute XY travel using plain G-code, independent of motion helpers.
    """
    g: List[str] = []
    g.append("G90")  # absolute
    if safe_lift and _DEF_SAFE_Z > 0:
        g += ["G91", f"G0 Z{_DEF_SAFE_Z:.3f} F{_DEF_FEED_Z}", "G90"]
    g.append(f"G0 X{float(x_mm):.3f} Y{float(y_mm):.3f} F{_DEF_FEED_XY}")
    return g

# -----------------------------------------------------------------------------
# UI & Templating
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "app_name": APP_NAME, "version": APP_VERSION})

# -----------------------------------------------------------------------------
# Camera endpoints
# -----------------------------------------------------------------------------
@app.get("/camera/stream")
def camera_stream():
    try:
        gen = camera.mjpeg_stream()  # yields multipart JPEG frames  :contentReference[oaicite:15]{index=15}
        return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(500, f"Camera stream error: {e}")

@app.post("/camera/capture")
def camera_capture():
    try:
        path = camera.capture()  # returns file path  :contentReference[oaicite:16]{index=16}
        capture_id = uuid.uuid4().hex
        CAPTURES[capture_id] = path
        try:
            models.record_capture(path)  # optional DB
        except Exception:
            pass
        return {"capture_id": capture_id, "path": path}
    except Exception as e:
        raise HTTPException(500, f"Capture error: {e}")

# -----------------------------------------------------------------------------
# OCR endpoints
# -----------------------------------------------------------------------------
@app.post("/ocr/{capture_id}")
def ocr_run(capture_id: str):
    path = CAPTURES.get(capture_id)
    if not path:
        raise HTTPException(404, "capture_id unknown")
    try:
        result = ocr.run(path)  # returns dict with text/conf  :contentReference[oaicite:17]{index=17}
        ocr_id = uuid.uuid4().hex
        OCR_RUNS[ocr_id] = result
        try:
            cap_row_id = models.record_capture(path)
            models.record_ocr(cap_row_id, result)
        except Exception:
            pass
        return {"ocr_id": ocr_id, **result}
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")

# -----------------------------------------------------------------------------
# Assignment / sort pipeline (capture → OCR → decide → move plan)
# -----------------------------------------------------------------------------
@app.post("/sort")
def sort_pipeline():
    # Capture
    try:
        path = camera.capture()
        capture_id = uuid.uuid4().hex
        CAPTURES[capture_id] = path
    except Exception as e:
        raise HTTPException(500, f"Capture error: {e}")

    # OCR
    try:
        ocr_res = ocr.run(path)
        ocr_id = uuid.uuid4().hex
        OCR_RUNS[ocr_id] = ocr_res
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")

    # Assign (simple mode; extend for game-specific rules on demand)
    try:
        assign.init(CONFIG.assignment or {})  # loads groups/csv if configured  :contentReference[oaicite:18]{index=18}
        decision = assign.decide(ocr_res.get("text", ""))
        slot_id = decision.get("slot_id")
    except Exception as e:
        raise HTTPException(500, f"Assign error: {e}")

    move_id = None
    if slot_id:
        # Convert to XY via GRID
        if not GRID:
            raise HTTPException(500, "GRID not initialized")
        r, c = GRID.slotid_to_rc(int(slot_id))
        x, y = GRID.slot_rc_to_xy(r, c)  # XY in mm  :contentReference[oaicite:19]{index=19}
        gcode = _plan_xy_move_gcode(x, y, safe_lift=True)
        try:
            motion.send(gcode)  # send G-code list to Marlin  :contentReference[oaicite:20]{index=20}
            move_id = uuid.uuid4().hex
        except Exception as e:
            raise HTTPException(500, f"Motion error: {e}")

    return {
        "capture_id": capture_id,
        "ocr_id": ocr_id,
        "assignment": decision,
        "move_id": move_id,
        "text": ocr_res.get("text"),
        "confidence": ocr_res.get("confidence"),
    }

# -----------------------------------------------------------------------------
# Motion endpoints
# -----------------------------------------------------------------------------
class GCodeBody(BaseModel):
    gcode: List[str]

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

@app.post("/motion/connect")
def motion_connect(port: str = "", baud: int = 0):
    try:
        return motion.connect(port=port or None, baud=baud or None)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/motion/disconnect")
def motion_disconnect():
    try:
        return motion.close()
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

@app.post("/motion/jog")
def motion_jog(axis: str, delta_mm: float, feed_xy: int = 1200):
    """
    Simple jog using relative move via G-code to avoid relying on helper availability.
    """
    axis = axis.upper().strip()
    if axis not in ("X", "Y", "Z"):
        raise HTTPException(400, "axis must be X, Y, or Z")
    try:
        g = ["G91", f"G0 {axis}{float(delta_mm):.3f} F{int(feed_xy)}", "G90"]
        return motion.send(g)
    except Exception as e:
        raise HTTPException(500, str(e))

# -----------------------------------------------------------------------------
# Grid endpoints (zone/letter/cell moves using planner)
# -----------------------------------------------------------------------------
@app.get("/grid/zones")
def grid_list_zones():
    if not GRID:
        raise HTTPException(500, "GRID not initialized")
    try:
        return {"zones": GRID.list_zones()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/zone")
def grid_move_zone(key: str, safe_lift: bool = True):
    if not GRID:
        raise HTTPException(500, "GRID not initialized")
    try:
        z = GRID.get_zone(key)  # resolve to XY  :contentReference[oaicite:21]{index=21}
        g = _plan_xy_move_gcode(z.x_mm, z.y_mm, safe_lift=safe_lift)
        res = motion.send(g)
        return {"ok": True, "move_id": uuid.uuid4().hex, "sent": g, "response": res}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/rc")
def grid_move_rc(row: int, col: int, safe_lift: bool = True):
    if not GRID:
        raise HTTPException(500, "GRID not initialized")
    try:
        x, y = GRID.slot_rc_to_xy(int(row), int(col))
        g = _plan_xy_move_gcode(x, y, safe_lift=safe_lift)
        res = motion.send(g)
        return {"ok": True, "move_id": uuid.uuid4().hex, "sent": g, "response": res}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/slotid")
def grid_move_slotid(slot_id: int, safe_lift: bool = True):
    if not GRID:
        raise HTTPException(500, "GRID not initialized")
    try:
        r, c = GRID.slotid_to_rc(int(slot_id))
        x, y = GRID.slot_rc_to_xy(r, c)
        g = _plan_xy_move_gcode(x, y, safe_lift=safe_lift)
        res = motion.send(g)
        return {"ok": True, "move_id": uuid.uuid4().hex, "sent": g, "response": res}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/grid/move/letter")
def grid_move_letter(letter: str, safe_lift: bool = True):
    if not GRID:
        raise HTTPException(500, "GRID not initialized")
    try:
        z = GRID.letter_to_zone(letter)  # letter → zone via alpha_map  :contentReference[oaicite:22]{index=22}
        if not z:
            raise HTTPException(404, f"No zone mapped for letter '{letter}'")
        g = _plan_xy_move_gcode(z.x_mm, z.y_mm, safe_lift=safe_lift)
        res = motion.send(g)
        return {"ok": True, "move_id": uuid.uuid4().hex, "sent": g, "response": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# -----------------------------------------------------------------------------
# Plunger / Vacuum endpoints
# -----------------------------------------------------------------------------
@app.post("/plunger/jog")
def plunger_jog(steps: int, enable: bool = True, delay_us: int = 1200):
    """
    Jog the auxiliary stepper (e.g., vacuum) by a signed number of steps.
    Positive: forward; Negative: reverse.
    """
    try:
        return plunger.jog(steps=steps, enable=enable, delay_us=int(delay_us))  # :contentReference[oaicite:23]{index=23}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/vacuum/on")
def vacuum_on():
    try:
        return plunger.jog(steps=400, enable=True, delay_us=1200)  # tweak steps for your pump  :contentReference[oaicite:24]{index=24}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/vacuum/off")
def vacuum_off():
    try:
        return plunger.jog(steps=-400, enable=True, delay_us=1200)  # reverse/stop  :contentReference[oaicite:25]{index=25}
    except Exception as e:
        raise HTTPException(500, str(e))
