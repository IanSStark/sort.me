# app/services/ocr.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger("ocr")

# =========================
# Module configuration/state
# =========================

@dataclass
class OCRState:
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 6
    whitelist: Optional[str] = None
    enable_orientation: bool = True
    enable_deskew: bool = True
    # Preprocessing toggles
    gamma: float = 1.0
    denoise: bool = True
    unsharp: bool = True
    adaptive_thresh: bool = True
    invert_if_needed: bool = True
    # Binarization fallback if adaptive fails
    otsu_fallback: bool = True

_state = OCRState()
_initialized = False


# =========================
# Public API (used by main)
# =========================

def init(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize OCR service. Safe to call multiple times.
    cfg keys expected (all optional):
      engine, lang, psm, whitelist
      enable_orientation, enable_deskew
      gamma, denoise, unsharp, adaptive_thresh, invert_if_needed, otsu_fallback
    """
    global _state, _initialized
    if cfg:
        _state.engine = cfg.get("engine", _state.engine)
        _state.lang = cfg.get("lang", _state.lang)
        _state.psm = int(cfg.get("psm", _state.psm))
        _state.whitelist = cfg.get("whitelist", _state.whitelist)

        _state.enable_orientation = bool(cfg.get("enable_orientation", _state.enable_orientation))
        _state.enable_deskew = bool(cfg.get("enable_deskew", _state.enable_deskew))

        _state.gamma = float(cfg.get("gamma", _state.gamma))
        _state.denoise = bool(cfg.get("denoise", _state.denoise))
        _state.unsharp = bool(cfg.get("unsharp", _state.unsharp))
        _state.adaptive_thresh = bool(cfg.get("adaptive_thresh", _state.adaptive_thresh))
        _state.invert_if_needed = bool(cfg.get("invert_if_needed", _state.invert_if_needed))
        _state.otsu_fallback = bool(cfg.get("otsu_fallback", _state.otsu_fallback))

    # Tesseract presence sanity check
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        logger.warning("Tesseract not found or not working: %s", e)

    _initialized = True
    logger.info("OCR initialized: engine=%s, lang=%s, psm=%s", _state.engine, _state.lang, _state.psm)


def status() -> bool:
    return _initialized


def run(img_path: str) -> Dict[str, Any]:
    """
    Execute OCR on the provided image path.
    Returns a dict: { 'text': str, 'confidence': int, 'boxes': dict }
    Raises RuntimeError on failure.
    """
    if not _initialized:
        raise RuntimeError("OCR service not initialized")

    path = Path(img_path)
    if not path.exists():
        raise RuntimeError(f"Image not found: {img_path}")

    # Load image
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image")

    # 1) Preprocess
    proc = _preprocess_for_ocr(bgr, _state)

    # 2) Optional orientation detection (Tesseract OSD)
    if _state.enable_orientation:
        angle = _estimate_rotation_angle(proc, _state)
        if angle and _state.enable_deskew:
            proc = _rotate_image(proc, -angle)

    # 3) OCR via Tesseract
    text, data = _run_tesseract(proc, _state)

    # 4) Confidence estimate (max line/word conf is typical; mean is often pessimistic)
    conf = _best_confidence_from_data(data)

    return {
        "text": text.strip(),
        "confidence": conf,
        "boxes": data,  # pytesseract image_to_data dict (words, conf, left/top/width/height, etc.)
    }


# =========================
# Preprocessing pipeline
# =========================

def _preprocess_for_ocr(bgr: np.ndarray, st: OCRState) -> np.ndarray:
    """
    Robust default preprocessing aimed at printed card titles/logos:
      - convert to grayscale
      - optional gamma adjust
      - light denoise
      - unsharp mask
      - adaptive threshold to improve contrast for OCR
      - auto-invert if background is dark
    Returns a single-channel uint8 image ready for OCR.
    """
    # To gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Gamma correction (handle glare/underexposure)
    if abs(st.gamma - 1.0) > 1e-3:
        gray = _apply_gamma(gray, st.gamma)

    # Light denoise
    if st.denoise:
        # Fast bilateral preserves edges better than gaussian for text
        try:
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        except Exception:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Unsharp mask (improves edge contrast for small fonts)
    if st.unsharp:
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Binarization
    bin_img = None
    if st.adaptive_thresh:
        try:
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 5
            )
        except Exception:
            bin_img = None

    if bin_img is None and st.otsu_fallback:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bin_img is None:
        bin_img = gray

    # Optional auto-invert (if background is dark and text is light)
    if st.invert_if_needed and _should_invert(bin_img):
        bin_img = cv2.bitwise_not(bin_img)

    return bin_img


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.05, min(5.0, gamma))
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


def _should_invert(bin_img: np.ndarray) -> bool:
    """
    Heuristic: if more than ~60% of pixels are dark, assume white text on dark bg → invert.
    """
    hist = cv2.calcHist([bin_img], [0], None, [2], [0, 256])  # 2-bin coarse hist
    dark = hist[0][0]
    total = bin_img.size
    return (dark / total) > 0.60


# =========================
# Orientation / Deskew
# =========================

def _estimate_rotation_angle(img: np.ndarray, st: OCRState) -> Optional[float]:
    """
    Use Tesseract OSD if available to estimate rotation.
    Returns angle in degrees (positive CCW) or None.
    """
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        # Tesseract typically reports rotation needed to correct to 0°
        angle = float(osd.get("rotate", 0.0))
        if abs(angle) < 0.5:
            return 0.0
        logger.debug("OSD rotation suggested: %.2f°", angle)
        return angle
    except Exception:
        # If OSD unavailable or fails, skip quietly
        return None


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if not angle_deg or abs(angle_deg) < 0.1:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Use border replicate to avoid black corners cutting text
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# =========================
# Tesseract invocation
# =========================

def _build_tesseract_config(st: OCRState) -> str:
    """
    Build a tesseract CLI config string based on state.
    Common PSMs:
      6 - Assume a single uniform block of text
      7 - Treat the image as a single text line
      11 - Sparse text with OSD
      13 - Raw line (no layout)
    """
    cfg_parts = [f"--psm {int(st.psm)}"]
    if st.whitelist:
        # tessedit_char_whitelist applies to legacy engine; still useful for filtering
        cfg_parts.append(f"-c tessedit_char_whitelist={st.whitelist}")
    # LSTM defaults are good for print
    return " ".join(cfg_parts)


def _run_tesseract(img: np.ndarray, st: OCRState) -> tuple[str, Dict[str, Any]]:
    config = _build_tesseract_config(st)
    try:
        text = pytesseract.image_to_string(img, lang=st.lang, config=config)
    except Exception as e:
        logger.error("Tesseract image_to_string failed: %s", e, exc_info=True)
        raise RuntimeError("OCR text extraction failed")

    try:
        data = pytesseract.image_to_data(img, lang=st.lang, config=config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        logger.error("Tesseract image_to_data failed: %s", e, exc_info=True)
        # Provide a minimal boxes structure if detailed data failed
        data = {"level": [], "text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}

    return text, data


def _best_confidence_from_data(data: Dict[str, Any]) -> int:
    """
    Produce a single 0–100 confidence score.
    Strategy: use the 75th percentile of valid word confidences to avoid outliers,
    then clamp to integer [0, 100].
    """
    try:
        confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
        if not confs:
            return 0
        confs.sort()
        idx = int(0.75 * (len(confs) - 1))
        val = confs[idx]
        return max(0, min(100, int(val)))
    except Exception:
        return 0
