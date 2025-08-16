# app/services/ocr.py
from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

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
    psm: int = 7                           # single line bias (top-name strip)
    whitelist: Optional[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'-. "

    # preprocessing
    enable_deskew: bool = True
    gamma: float = 1.1
    denoise: bool = True
    unsharp: bool = True
    adaptive_thresh: bool = True
    invert_if_needed: bool = True
    otsu_fallback: bool = True

    # orientation attempts (we explicitly want 0 and 180)
    rotations: Tuple[int, int] = (0, 180)

    # upscale small ROIs so OCR has enough pixels
    min_ocr_width: int = 900

    # fixed policy: OCR only the top third of the (possibly rotated) image
    top_fraction: float = 1.0 / 3.0

    initialized: bool = False


_state = OCRState()


# =========================
# Public API
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize OCR module. We always crop the *top third* after rotation.
    Optional config keys honored:
      engine, lang, psm, whitelist,
      enable_deskew, gamma, denoise, unsharp, adaptive_thresh, invert_if_needed, otsu_fallback,
      rotations (e.g., [0, 180]), min_ocr_width
    """
    global _state
    _state.engine = str(cfg.get("engine", _state.engine))
    _state.lang = str(cfg.get("lang", _state.lang))
    _state.psm = int(cfg.get("psm", _state.psm))

    wl = cfg.get("whitelist", _state.whitelist)
    _state.whitelist = None if wl in (None, "", "null", "None") else str(wl)

    _state.enable_deskew = bool(cfg.get("enable_deskew", _state.enable_deskew))
    _state.gamma = float(cfg.get("gamma", _state.gamma))
    _state.denoise = bool(cfg.get("denoise", _state.denoise))
    _state.unsharp = bool(cfg.get("unsharp", _state.unsharp))
    _state.adaptive_thresh = bool(cfg.get("adaptive_thresh", _state.adaptive_thresh))
    _state.invert_if_needed = bool(cfg.get("invert_if_needed", _state.invert_if_needed))
    _state.otsu_fallback = bool(cfg.get("otsu_fallback", _state.otsu_fallback))

    rots = cfg.get("rotations", _state.rotations)
    if isinstance(rots, (list, tuple)) and rots:
        _state.rotations = tuple(int(r) for r in rots)

    _state.min_ocr_width = int(cfg.get("min_ocr_width", _state.min_ocr_width))

    # Validate tesseract availability
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
        "OCR initialized: engine=%s, lang=%s, psm=%d, rotations=%s, policy=top-third",
        _state.engine, _state.lang, _state.psm, _state.rotations
    )


def status() -> bool:
    return _state.initialized


def run(image_path: str) -> Dict[str, Any]:
    """
    Read the entire image, evaluate 0° and 180°, then OCR the *top third* of the rotated image.
    Returns:
      {
        'text': str,
        'confidence': int,
        'boxes': {...},
        'rotation': int,       # 0 or 180
        'roi_px': [x1,y1,x2,y2]
      }
    """
    if not _state.initialized:
        raise RuntimeError("OCR not initialized")

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    best = {"confidence": -1, "text": "", "data": {}, "rotation": 0, "roi": (0, 0, 0, 0)}

    for rot in _state.rotations:
        rotated = _rotate_bound_color(img_bgr, rot) if rot != 0 else img_bgr
        # Crop top third after rotation
        hr, wr = rotated.shape[:2]
        x1, y1, x2, y2 = _top_third_rect_px(wr, hr, _state.top_fraction)
        roi = rotated[y1:y2, x1:x2].copy()

        # Preprocess ROI for OCR
        proc = _preprocess(roi, _state)

        # Tesseract
        text, data = _run_tesseract(proc, _state)

        # Choose the "best" line candidate
        text_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pick = max(text_lines, key=lambda s: (_alpha_ratio(s), len(s))) if text_lines else text.strip()
        conf = int(max(0, min(100, data.get("confidence", 0))))

        # prefer higher confidence; break ties by alpha ratio + length
        tie_cur = (_alpha_ratio(pick), conf)
        tie_best = (_alpha_ratio(best["text"]), best["confidence"])
        if (conf > best["confidence"]) or (conf == best["confidence"] and tie_cur > tie_best):
            best = {"confidence": conf, "text": pick, "data": data, "rotation": rot, "roi": (x1, y1, x2, y2)}

    result = {
        "text": best["text"].replace("\r", " ").strip(),
        "confidence": int(best["confidence"]),
        "boxes": best["data"].get("boxes", {}),
        "rotation": int(best["rotation"]),
        "roi_px": list(best.get("roi", (0, 0, 0, 0))),
    }
    return result


# =========================
# Core steps
# =========================

def _preprocess(img_bgr: np.ndarray, st: OCRState) -> np.ndarray:
    """
    Prepare ROI for robust single-line OCR.
    """
    img = img_bgr.copy()

    # Ensure sufficient width for OCR
    h, w = img.shape[:2]
    if w < st.min_ocr_width:
        scale = st.min_ocr_width / float(max(1, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Gamma correction
    if st.gamma and abs(st.gamma - 1.0) > 1e-3:
        try:
            inv_gamma = 1.0 / max(1e-3, st.gamma)
            table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
            table = np.clip(table, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, table)
        except Exception:
            logger.debug("Gamma correction skipped", exc_info=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoise
    if st.denoise:
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Deskew
    if st.enable_deskew:
        try:
            angle = _estimate_skew_angle(gray)
            if abs(angle) > 0.5:
                gray = _rotate_bound(gray, -angle)
        except Exception:
            logger.debug("Deskew skipped", exc_info=True)

    # Unsharp masking
    if st.unsharp:
        try:
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        except Exception:
            logger.debug("Unsharp skip", exc_info=True)

    # Binarize
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

    # Optional inversion if background too bright
    if st.invert_if_needed:
        try:
            if float(np.mean(bin_img)) > 180:
                bin_img = cv2.bitwise_not(bin_img)
        except Exception:
            pass

    return bin_img


def _run_tesseract(img: np.ndarray, st: OCRState) -> Tuple[str, Dict[str, Any]]:
    """
    Run tesseract with single-line bias and optional whitelist.
    """
    tokens: List[str] = [f"--psm {int(getattr(st, 'psm', 7))}"]
    wl = getattr(st, "whitelist", None)
    if wl:
        tokens.append(f"-c tessedit_char_whitelist={_quote_tess_value(str(wl))}")
    config = " ".join(tokens)

    try:
        text = pytesseract.image_to_string(img, lang=st.lang, config=config)

        # Collect conf/boxes (word level)
        data: Dict[str, Any] = {}
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
            data = {"boxes": {}, "confidence": _estimate_confidence(text)}

        return text, data

    except Exception as e:
        logger.error("Tesseract failed: %s", e, exc_info=True)
        raise RuntimeError("OCR text extraction failed")


# =========================
# Utilities
# =========================

def _top_third_rect_px(w: int, h: int, top_fraction: float) -> Tuple[int, int, int, int]:
    """Return rectangle covering the top `top_fraction` of the image."""
    top_fraction = max(0.05, min(0.5, float(top_fraction)))  # guardrails: between 5% and 50%
    y2 = int(round(h * top_fraction))
    return 0, 0, w, max(1, y2)

def _quote_tess_value(val: str) -> str:
    # Safe for shlex in pytesseract
    return '"' + val.replace('\\', '\\\\').replace('"', '\\"') + '"'

def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    alnum = sum(ch.isalnum() for ch in s)
    return alnum / max(1, len(s))

def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180.0, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines:
        _, theta = rho_theta[0]
        angle = (theta * 180.0 / np.pi) - 90.0
        if angle < -90: angle += 180
        if angle >  90: angle -= 180
        if -45 <= angle <= 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))

def _rotate_bound(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0,0]), abs(m[0,1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    m[0, 2] += (nW / 2) - center[0]
    m[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(gray, m, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _rotate_bound_color(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0,0]), abs(m[0,1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    m[0, 2] += (nW / 2) - center[0]
    m[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, m, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _aggregate_confidence(confs: List[int], text: str) -> int:
    vals = [c for c in confs if isinstance(c, (int, float)) and c >= 0]
    if not vals:
        return _estimate_confidence(text)
    return int(round(float(np.mean(vals))))

def _estimate_confidence(text: str) -> int:
    if not text or not text.strip():
        return 0
    t = text.strip()
    letters = sum(ch.isalnum() for ch in t)
    ratio = letters / max(1, len(t))
    base = 35 + int(60 * ratio)
    length_bonus = 0
    try:
        length_bonus = min(10, int(math.log10(max(10, len(t))) * 8))
    except Exception:
        pass
    return int(max(0, min(100, base + length_bonus)))
