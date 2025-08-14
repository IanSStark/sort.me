# app/services/ocr.py
from __future__ import annotations

import logging
import math
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger("ocr")

# =========================
# State / configuration
# =========================

@dataclass
class OCRState:
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 6
    whitelist: Optional[str] = None

    # Preprocessing toggles
    enable_orientation: bool = True
    enable_deskew: bool = True
    gamma: float = 1.0
    denoise: bool = True
    unsharp: bool = True
    adaptive_thresh: bool = True
    invert_if_needed: bool = True
    otsu_fallback: bool = True

    initialized: bool = False


_state = OCRState()


# =========================
# Public API
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize the OCR module with configuration from main.py / config.yaml.
    """
    global _state
    _state.engine = str(cfg.get("engine", _state.engine))
    _state.lang = str(cfg.get("lang", _state.lang))
    _state.psm = int(cfg.get("psm", _state.psm))
    wl = cfg.get("whitelist", _state.whitelist)
    _state.whitelist = None if wl in (None, "", "null", "None") else str(wl)

    _state.enable_orientation = bool(cfg.get("enable_orientation", _state.enable_orientation))
    _state.enable_deskew = bool(cfg.get("enable_deskew", _state.enable_deskew))
    _state.gamma = float(cfg.get("gamma", _state.gamma))
    _state.denoise = bool(cfg.get("denoise", _state.denoise))
    _state.unsharp = bool(cfg.get("unsharp", _state.unsharp))
    _state.adaptive_thresh = bool(cfg.get("adaptive_thresh", _state.adaptive_thresh))
    _state.invert_if_needed = bool(cfg.get("invert_if_needed", _state.invert_if_needed))
    _state.otsu_fallback = bool(cfg.get("otsu_fallback", _state.otsu_fallback))

    # Basic sanity check that tesseract is callable
    if _state.engine.lower() == "tesseract":
        try:
            out = subprocess.run(
                ["tesseract", "--version"], capture_output=True, text=True, check=False
            )
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or "tesseract not available")
        except FileNotFoundError:
            raise RuntimeError("tesseract binary not found; install tesseract-ocr")

    _state.initialized = True
    logger.info(
        "OCR initialized: engine=%s, lang=%s, psm=%d",
        _state.engine, _state.lang, _state.psm
    )


def status() -> bool:
    return _state.initialized


def run(image_path: str) -> Dict[str, Any]:
    """
    Execute OCR on the given image path.
    Returns: { 'text': str, 'confidence': int, 'boxes': {...} }
    """
    if not _state.initialized:
        raise RuntimeError("OCR not initialized")

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    proc = _preprocess(img_bgr, _state)
    text, data = _run_tesseract(proc, _state)

    # Clean final text
    text = text.replace("\r", " ").strip()

    # Confidence normalization to 0..100 (int)
    confidence = int(max(0, min(100, data.get("confidence", 0))))

    return {"text": text, "confidence": confidence, "boxes": data.get("boxes", {})}


# =========================
# Core steps
# =========================

def _preprocess(img_bgr: np.ndarray, st: OCRState) -> np.ndarray:
    """
    Apply a conservative, OCR-friendly preprocessing pipeline.
    Returns a single-channel or three-channel image suitable for Tesseract.
    """
    img = img_bgr.copy()

    # Optional gamma correction (improves contrast in low light)
    if st.gamma and abs(st.gamma - 1.0) > 1e-3:
        try:
            inv_gamma = 1.0 / max(1e-3, st.gamma)
            table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
            table = np.clip(table, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, table)
        except Exception:
            logger.debug("Gamma correction skipped", exc_info=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoise to reduce speckle, preserve edges
    if st.denoise:
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Optional orientation detection (coarse)
    if st.enable_orientation:
        try:
            angle = _estimate_skew_angle(gray)
            if abs(angle) > 0.5 and st.enable_deskew:
                gray = _rotate_bound(gray, -angle)
        except Exception:
            logger.debug("Orientation/deskew skipped", exc_info=True)

    # Unsharp mask to sharpen titles
    if st.unsharp:
        try:
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        except Exception:
            logger.debug("Unsharp skip", exc_info=True)

    # Binarization
    bin_img = None
    if st.adaptive_thresh:
        try:
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
        except Exception:
            logger.debug("Adaptive threshold failed", exc_info=True)

    if bin_img is None and st.otsu_fallback:
        try:
            _thr, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            logger.debug("OTSU threshold failed", exc_info=True)
            bin_img = gray

    # Optional inversion if text appears light on dark
    if st.invert_if_needed:
        try:
            mean_val = float(np.mean(bin_img))
            if mean_val > 180:  # very bright -> likely inverted
                bin_img = cv2.bitwise_not(bin_img)
        except Exception:
            pass

    # Tesseract accepts single channel or 3-channel BGR; prefer single channel.
    return bin_img


def _run_tesseract(img: np.ndarray, st: OCRState) -> Tuple[str, Dict[str, Any]]:
    """
    Run tesseract safely. Properly quotes -c values to avoid shlex errors.
    Returns (text, meta) where meta includes 'confidence' and 'boxes'.
    """
    # Build robust config string
    tokens = [f"--psm {int(getattr(st, 'psm', 6))}"]

    wl = getattr(st, "whitelist", None)
    if wl:
        tokens.append(f"-c tessedit_char_whitelist={_quote_tess_value(str(wl))}")

    config = " ".join(tokens)

    try:
        # Main text
        text = pytesseract.image_to_string(img, lang=st.lang, config=config)

        # Try to collect per-word boxes and confidences
        data = {}
        try:
            df = pytesseract.image_to_data(img, lang=st.lang, config=config, output_type=Output.DICT)
            boxes = []
            confs = []
            n = len(df.get("text", []))
            for i in range(n):
                t = (df["text"][i] or "").strip()
                conf = int(df["conf"][i]) if df["conf"][i] not in ("-1", "", None) else -1
                if t:
                    boxes.append({
                        "text": t,
                        "conf": conf,
                        "left": int(df["left"][i]),
                        "top": int(df["top"][i]),
                        "width": int(df["width"][i]),
                        "height": int(df["height"][i]),
                    })
                confs.append(conf)
            data["boxes"] = {"words": boxes}
            data["confidence"] = _aggregate_confidence(confs, text)

        except Exception:
            # Fallback if image_to_data fails
            data = {"boxes": {}, "confidence": _estimate_confidence(text)}

        return text, data

    except Exception as e:
        logger.error("Tesseract image_to_string failed: %s", e, exc_info=True)
        raise RuntimeError("OCR text extraction failed")


# =========================
# Utilities
# =========================

def _quote_tess_value(val: str) -> str:
    """
    Quote a value for use in `-c key=value` so that pytesseract's shlex.split()
    parses it safely. Use double quotes and escape \ and " (do not use single quotes).
    """
    return '"' + val.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate text skew angle via Hough on edges. Returns degrees.
    Positive angle means clockwise skew.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=120)
    if lines is None:
        return 0.0

    angles = []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        # Convert to degrees relative to horizontal text
        angle = (theta * 180.0 / np.pi) - 90.0
        # Normalize to [-45, 45] for stability
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        if -45 <= angle <= 45:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use median for robustness
    return float(np.median(angles))


def _rotate_bound(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate an image without cropping, keeping full content.
    """
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    m[0, 2] += (nW / 2) - center[0]
    m[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(gray, m, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _aggregate_confidence(confs: list[int], text: str) -> int:
    """
    Aggregate confidences from Tesseract's word list.
    Returns an int in 0..100. Ignores -1 entries.
    """
    vals = [c for c in confs if isinstance(c, (int, float)) and c >= 0]
    if not vals:
        return _estimate_confidence(text)
    # Weighted by sqrt(length) of reconstructed text might be overkill; simple mean is OK.
    return int(round(float(np.mean(vals))))


def _estimate_confidence(text: str) -> int:
    """
    Heuristic confidence if detailed confidences are missing.
    Penalize very short or whitespace-heavy outputs.
    """
    if not text or not text.strip():
        return 0
    t = text.strip()
    letters = sum(ch.isalnum() for ch in t)
    ratio = letters / max(1, len(t))
    # map ratio to [30..95] depending on length
    base = 30 + int(65 * ratio)
    length_bonus = min(10, int(math.log10(max(10, len(t))) * 8))
    return int(max(0, min(100, base + length_bonus)))

import pytesseract
import logging
from PIL import Image

logger = logging.getLogger("ocr")

def run_ocr(image_path: str) -> str:
    """Run OCR on the given image and return the detected text."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        logger.info(f"[OCR] Detected text: {text.strip()}")
        return text.strip()
    except Exception as e:
        logger.exception(f"OCR failed: {e}")
        return ""

