# app/services/camera.py
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2  # OpenCV is used for encoding/saving even with Picamera2

logger = logging.getLogger("camera")

# ---------- Module state ----------
_CAPTURES_DIR = Path("captures")
_BACKEND = None  # "picamera2" | "opencv" | None
_LOCK = threading.RLock()

# Backends
_picam2 = None          # type: ignore
_picam2_started = False
_opencv_cap = None      # type: Optional[cv2.VideoCapture]

# Configuration cache
_cfg_device: Optional[Union[int, str]] = None
_cfg_resolution: Tuple[int, int] = (1920, 1080)
_cfg_preview_fps: int = 5

# JPEG params
_JPEG_QUALITY = 92  # reasonable default for clarity vs size


# ---------- Public API expected by main.py ----------

def init(
    device: Optional[str] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    preview_fps: int = 5,
) -> None:
    """
    Initialize camera service.

    Args:
        device: If provided and looks like /dev/video* or an integer string, force OpenCV/V4L2.
                If None, prefer Picamera2 when available; fallback to OpenCV(0).
        resolution: Desired (width, height).
        preview_fps: Target FPS for preview streams (best effort).
    """
    global _BACKEND, _picam2, _picam2_started, _opencv_cap
    global _cfg_device, _cfg_resolution, _cfg_preview_fps

    with _LOCK:
        if _BACKEND is not None:
            logger.info("camera.init() called but backend already initialized: %s", _BACKEND)
            return

        _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

        _cfg_device = _coerce_device(device)
        _cfg_resolution = resolution
        _cfg_preview_fps = preview_fps

        # Try to use Picamera2 unless the device explicitly forces OpenCV
        if _should_try_picamera2(_cfg_device):
            try:
                from picamera2 import Picamera2  # type: ignore
                _picam2 = Picamera2()
                _configure_picamera2(_picam2, resolution, preview_fps)
                _picam2.start()
                _picam2_started = True
                _BACKEND = "picamera2"
                logger.info("Picamera2 backend initialized at %sx%s", *resolution)
                return
            except Exception as e:
                logger.warning("Picamera2 init failed, falling back to OpenCV: %s", e, exc_info=True)

        # OpenCV fallback
        _opencv_cap = _open_opencv_capture(_cfg_device, resolution, preview_fps)
        _BACKEND = "opencv"
        logger.info("OpenCV backend initialized on device=%s at %sx%s", _cfg_device, *resolution)


def status() -> bool:
    """Return True if the camera backend is initialized and ready."""
    with _LOCK:
        if _BACKEND == "picamera2":
            return bool(_picam2_started and _picam2 is not None)
        if _BACKEND == "opencv":
            return bool(_opencv_cap is not None and _opencv_cap.isOpened())
        return False


def capture() -> str:
    """
    Capture a single JPEG and return the absolute path as a string.

    Raises:
        RuntimeError if capturing fails or camera not initialized.
    """
    with _LOCK:
        if _BACKEND is None:
            raise RuntimeError("Camera not initialized")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = (_CAPTURES_DIR / f"{ts}.jpg").resolve()

        if _BACKEND == "picamera2":
            return _capture_picamera2(out_path)

        if _BACKEND == "opencv":
            return _capture_opencv(out_path)

        raise RuntimeError("Unknown camera backend")


def shutdown() -> None:
    """Release resources safely."""
    global _BACKEND, _picam2_started, _picam2, _opencv_cap
    with _LOCK:
        if _BACKEND == "picamera2":
            try:
                if _picam2 is not None and _picam2_started:
                    _picam2.stop()
            except Exception:
                logger.warning("Error stopping Picamera2", exc_info=True)
            finally:
                _picam2_started = False
                _picam2 = None

        if _BACKEND == "opencv":
            try:
                if _opencv_cap is not None:
                    _opencv_cap.release()
            except Exception:
                logger.warning("Error releasing OpenCV capture", exc_info=True)
            finally:
                _opencv_cap = None

        _BACKEND = None
        logger.info("Camera shutdown complete")


# ---------- Backend helpers ----------

def _coerce_device(device: Optional[str]) -> Optional[Union[int, str]]:
    if device is None:
        return None
    # Accept integer-like strings as indices
    try:
        return int(device)
    except (TypeError, ValueError):
        return device  # e.g., "/dev/video2"


def _should_try_picamera2(device: Optional[Union[int, str]]) -> bool:
    """Prefer Picamera2 if no explicit device override pointing to V4L2."""
    if device is None:
        return True
    # If user specified a /dev/video* or a numeric index, assume V4L2/OpenCV
    if isinstance(device, int):
        return False
    return not str(device).startswith("/dev/video")


def _configure_picamera2(p2, resolution: Tuple[int, int], preview_fps: int) -> None:
    """
    Configure Picamera2 for still capture with the requested resolution.
    Uses a raw mode suitable for text clarity and stable exposure.
    """
    # Create a simple still configuration
    cfg = p2.create_still_configuration(
        main={"size": (int(resolution[0]), int(resolution[1]))},
        buffer_count=2
    )
    p2.configure(cfg)

    # Optional: conservative controls for OCR clarity
    try:
        # Lock AWB/AE once settled to reduce flicker across captures
        p2.set_controls({"AwbEnable": True, "AeEnable": True})
        time.sleep(0.2)  # allow sensors to settle a bit
    except Exception:
        # Controls vary by driver; non-fatal if unsupported
        logger.debug("Some Picamera2 controls not supported", exc_info=True)


def _open_opencv_capture(
    device: Optional[Union[int, str]],
    resolution: Tuple[int, int],
    preview_fps: int,
) -> cv2.VideoCapture:
    index = 0 if device is None else device
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device: {index}")

    # Try to set resolution and fps
    w, h = int(resolution[0]), int(resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, int(preview_fps))

    # Warm up: read a few frames
    for _ in range(3):
        cap.read()
        time.sleep(0.03)

    # Verify capture
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("OpenCV camera test frame failed")

    # If the driver ignored the requested size, log but proceed
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (w, h):
        logger.warning("Requested %sx%s but got %sx%s", w, h, actual_w, actual_h)

    return cap


def _capture_picamera2(out_path: Path) -> str:
    assert _picam2 is not None
    try:
        # Capture to numpy array and encode via OpenCV for consistent JPEG params
        frame = _picam2.capture_array()
        if frame is None:
            raise RuntimeError("Picamera2 returned empty frame")

        # Convert if necessary (Picamera2 can produce RGB; OpenCV expects BGR)
        if frame.shape[-1] == 3:  # RGB -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out_path.write_bytes(buf.tobytes())
        logger.info("Captured image (Picamera2): %s", out_path)
        return str(out_path)
    except Exception as e:
        logger.error("Picamera2 capture failed: %s", e, exc_info=True)
        raise


def _capture_opencv(out_path: Path) -> str:
    assert _opencv_cap is not None
    # Grab a couple frames to reduce motion blur and allow auto-exposure to settle
    for _ in range(2):
        _opencv_cap.read()
        time.sleep(0.01)

    ok, frame = _opencv_cap.read()
    if not ok or frame is None:
        raise RuntimeError("OpenCV capture failed")

    # Optional: simple sharpening for OCR clarity (light unsharp mask)
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    except Exception:
        # If OpenCV build lacks some filters, skip silently
        pass

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    out_path.write_bytes(buf.tobytes())
    logger.info("Captured image (OpenCV): %s", out_path)
    return str(out_path)
