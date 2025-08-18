# app/services/camera.py
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

logger = logging.getLogger("camera")

# =========================
# Module state
# =========================
_CAPTURES_DIR = Path("captures")
_BACKEND: Optional[str] = None          # "picamera2" | "opencv" | None
_LOCK = threading.RLock()

# Picamera2 runtime
_picam2 = None                          # type: ignore
_picam2_started = False

# OpenCV runtime
_opencv_cap: Optional[cv2.VideoCapture] = None

# Cached configuration
_cfg_device: Optional[Union[int, str]] = None
_cfg_resolution: Tuple[int, int] = (1280, 720)
_cfg_preview_fps: int = 5

# JPEG quality (0–100)
_JPEG_QUALITY = 92


# =========================
# Public API (used by main.py)
# =========================

def init(
    device: Optional[str] = None,
    backend: Optional[str] = None,
    resolution: Tuple[int, int] = (1280, 720),
    captures_dir: Optional[str] = None,
    preview_fps: int = 5,
) -> None:
    """
    Initialize the camera service.

    Args:
        device: If None, prefer Picamera2; else force OpenCV/V4L2.
                Accepts '/dev/videoX' or an integer index as a string.
        backend: Optional explicit backend: "picamera2" or "opencv".
        resolution: (width, height) request.
        captures_dir: directory to store captured stills.
        preview_fps: Target FPS for preview/streaming (best effort).
    """
    global _BACKEND, _picam2, _picam2_started, _opencv_cap
    global _cfg_device, _cfg_resolution, _cfg_preview_fps, _CAPTURES_DIR

    with _LOCK:
        if _BACKEND is not None:
            logger.info("camera.init(): backend already initialized: %s", _BACKEND)
            return

        if captures_dir:
            _CAPTURES_DIR = Path(captures_dir)

        _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        _cfg_device = _coerce_device(device)
        _cfg_resolution = (int(resolution[0]), int(resolution[1]))
        _cfg_preview_fps = int(preview_fps)

        # Decide backend
        forced = (backend or "").strip().lower() or None

        if forced == "picamera2" or (forced is None and _should_try_picamera2(_cfg_device)):
            try:
                from picamera2 import Picamera2  # type: ignore
                _picam2 = Picamera2()
                _configure_picamera2(_picam2, _cfg_resolution, _cfg_preview_fps)
                _picam2.start()
                _picam2_started = True
                _BACKEND = "picamera2"
                logger.info("Picamera2 backend initialized at %sx%s", *_cfg_resolution)
                return
            except Exception as e:
                logger.warning("Picamera2 init failed, falling back to OpenCV: %s", e, exc_info=True)

        # OpenCV fallback via V4L2
        _opencv_cap = _open_opencv_capture(_cfg_device, _cfg_resolution, _cfg_preview_fps)
        _BACKEND = "opencv"
        logger.info("OpenCV backend initialized on device=%s at %sx%s", _cfg_device, *_cfg_resolution)


def status() -> bool:
    """Return True if the camera backend is initialized and ready."""
    with _LOCK:
        if _BACKEND == "picamera2":
            return bool(_picam2 is not None and _picam2_started)
        if _BACKEND == "opencv":
            return bool(_opencv_cap is not None and _opencv_cap.isOpened())
        return False


def capture() -> str:
    """
    Capture a single JPEG to disk. Returns absolute file path.
    Raises RuntimeError on failure.
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


def get_frame():
    """
    Return a single BGR frame (numpy ndarray) or None.
    Used by the stream fallback when JPEG bytes aren't provided directly.
    """
    with _LOCK:
        if _BACKEND == "picamera2":
            if _picam2 is None:
                return None
            frame = _picam2.capture_array()
            if frame is None:
                return None
            # Picamera2 gives RGB; convert to BGR for OpenCV/JPEG encode
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame

        if _BACKEND == "opencv" and _opencv_cap is not None:
            # small warm-up to stabilize auto-exposure
            _opencv_cap.read()
            ok, frame = _opencv_cap.read()
            return frame if ok else None

        return None


def get_jpeg_frame() -> Optional[bytes]:
    """
    Return a single JPEG frame (bytes) without saving to disk, or None on failure.
    Preferred by the MJPEG route to avoid re-encoding work elsewhere.
    """
    frame = get_frame()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        return None
    return bytes(buf)


def mjpeg_stream():
    """
    Native MJPEG generator. Each yielded chunk is a valid multipart part:
      --frame\r\n
      Content-Type: image/jpeg\r\n
      Content-Length: <n>\r\n
      \r\n
      <JPEG bytes>\r\n
    """
    boundary = b"--frame"
    while True:
        try:
            jpg = get_jpeg_frame()
            if not jpg:
                time.sleep(0.05)
                continue

            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                b"\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.05)  # ~20 fps target
        except GeneratorExit:
            break
        except Exception:
            # transient issue; keep streaming
            time.sleep(0.2)


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


# =========================
# Backend helpers
# =========================

def _coerce_device(device: Optional[str]) -> Optional[Union[int, str]]:
    if device is None:
        return None
    # Accept integer-like strings as indices
    try:
        return int(device)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return device  # e.g., "/dev/video0"


def _should_try_picamera2(device: Optional[Union[int, str]]) -> bool:
    """
    Prefer Picamera2 if no explicit V4L2 device/index was provided.
    """
    if device is None:
        return True
    if isinstance(device, int):
        return False
    return not str(device).startswith("/dev/video")


def _configure_picamera2(p2, resolution: Tuple[int, int], preview_fps: int) -> None:
    """
    Configure Picamera2 for still/preview at the requested resolution.
    """
    w, h = int(resolution[0]), int(resolution[1])
    cfg = p2.create_still_configuration(main={"size": (w, h)}, buffer_count=2)
    p2.configure(cfg)
    # Conservative control setup; allow AWB/AE to settle a moment.
    try:
        p2.set_controls({"AwbEnable": True, "AeEnable": True})
        time.sleep(0.2)
    except Exception:
        logger.debug("Some Picamera2 controls not supported", exc_info=True)


def _open_opencv_capture(
    device: Optional[Union[int, str]],
    resolution: Tuple[int, int],
    preview_fps: int,
) -> cv2.VideoCapture:
    """
    Open a V4L2 device robustly. Request MJPG, warm up, and verify a test frame.
    """
    index = 0 if device is None else device
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device: {index}")

    w, h = int(resolution[0]), int(resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, int(preview_fps))
    # Many UVC devices prefer MJPG for higher resolutions
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    # Warm-up & retry loop
    ok, frame = False, None
    for _ in range(10):
        _ = cap.read()
        time.sleep(0.05)
        ok, frame = cap.read()
        if ok and frame is not None:
            break

    if not ok or frame is None:
        cap.release()
        raise RuntimeError("OpenCV camera test frame failed")

    # Log actual negotiated resolution if driver adjusted it
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (w, h):
        logger.warning("Requested %sx%s but got %sx%s", w, h, actual_w, actual_h)

    return cap


def _capture_picamera2(out_path: Path) -> str:
    assert _picam2 is not None
    try:
        frame = _picam2.capture_array()
        if frame is None:
            raise RuntimeError("Picamera2 returned empty frame")

        # Picamera2 frames are typically RGB; encode with OpenCV (expects BGR)
        if frame.shape[-1] == 3:
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
    # Grab a couple frames to stabilize exposure
    for _ in range(2):
        _opencv_cap.read()
        time.sleep(0.01)

    ok, frame = _opencv_cap.read()
    if not ok or frame is None:
        raise RuntimeError("OpenCV capture failed")

    # Mild unsharp mask to aid OCR
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    except Exception:
        pass

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    out_path.write_bytes(buf.tobytes())
    logger.info("Captured image (OpenCV): %s", out_path)
    return str(out_path)
