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
    # default single-line bias; we will also try other PSMs in an ensemble
    psm: int = 7
    # keep whitelist optional; set to None for max flexibility
    whitelist: Optional[str] = None  # e.g. "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'-. "

    # preprocessing
    enable_deskew: bool = True        # small-angle deskew only (not 180° flips)
    gamma: float = 1.1
    denoise: bool = True
    unsharp: bool = True
    adaptive_thresh: bool = True
    invert_if_needed: bool = True
    otsu_fallback: bool = True

    # upscale small ROIs so OCR has enough pixels
    min_ocr_width: int = 900

    # policy: OCR only the top third of the (non-rotated) image
    top_fraction: float = 1.0 / 3.0

    initialized: bool = False


_state = OCRState()


# =========================
# Public API
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize OCR module. We always crop the *top third* (no orientation flips).
    Optional config keys honored:
      engine, lang, psm, whitelist,
      enable_deskew, gamma, denoise, unsharp, adaptive_thresh, invert_if_needed, otsu_fallback,
      min_ocr_width
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
        "OCR initialized: engine=%s, lang=%s, psm=%d, policy=top-third (no 0/180 rotation)",
        _state.engine, _state.lang, _state.psm
    )


def status() -> bool:
    return _state.initialized


def run(image_path: str) -> Dict[str, Any]:
    """
    Read the image, crop the *top third*, preprocess, run an OCR ensemble (no orientation flips).
    Returns:
      {
        'text': str,
        'confidence': int,     # 0..100
        'boxes': {...},        # word boxes
        'rotation': 0,         # fixed (no orientation attempts)
        'roi_px': [x1,y1,x2,y2]
      }
    """
    if not _state.initialized:
        raise RuntimeError("OCR not initialized")

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _top_third_rect_px(w, h, _state.top_fraction)
    roi_bgr = img_bgr[y1:y2, x1:x2].copy()

    # Preprocess baseline (grayscale pipeline with deskew, etc.)
    base = _preprocess(roi_bgr, _state)

    # Build variants for ensemble
    variants: List[Tuple[str, np.ndarray]] = [("base", base)]

    # Additional binarizations to try (if not already binarized by preprocess stage):
    # We will regenerate from a pre-deskewed grayscale to keep consistency.
    gray_for_alts = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if _state.denoise:
        gray_for_alts = cv2.bilateralFilter(gray_for_alts, d=7, sigmaColor=50, sigmaSpace=50)
    if _state.enable_deskew:
        try:
            ang = _estimate_skew_angle(gray_for_alts)
            if abs(ang) > 0.5:
                gray_for_alts = _rotate_bound(gray_for_alts, -ang)
        except Exception:
            logger.debug("Alt deskew skipped", exc_info=True)
    if _state.unsharp:
        try:
            blur = cv2.GaussianBlur(gray_for_alts, (0, 0), 1.0)
            gray_for_alts = cv2.addWeighted(gray_for_alts, 1.6, blur, -0.6, 0)
        except Exception:
            pass

    try:
        adp = cv2.adaptiveThreshold(gray_for_alts, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        variants.append(("adaptive", adp))
    except Exception:
        pass

    try:
        _thr, otsu = cv2.threshold(gray_for_alts, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", otsu))
    except Exception:
        pass

    # Optional inversion if background very bright (applied to each variant copy)
    def maybe_invert(img: np.ndarray) -> np.ndarray:
        if _state.invert_if_needed:
            try:
                if float(np.mean(img)) > 180:
                    return cv2.bitwise_not(img)
            except Exception:
                pass
        return img

    # PSMs to try (single line, block, sparse text)
    psms = [int(_state.psm), 6, 11]
    seen_psm = set()
    psms = [p for p in psms if not (p in seen_psm or seen_psm.add(p))]

    # Whitelist strategies: try with and without (if provided)
    whitelists: List[Optional[str]] = [None]
    if _state.whitelist:
        whitelists.append(_state.whitelist)

    best = {"score": -1.0, "text": "", "data": {}, "confidence": 0, "tag": ""}

    for tag, var in variants:
        img = maybe_invert(var)
        # Ensure min width for Tesseract
        hh, ww = img.shape[:2]
        if ww < _state.min_ocr_width:
            scale = _state.min_ocr_width / float(max(1, ww))
            img = cv2.resize(img, (int(ww * scale), int(hh * scale)), interpolation=cv2.INTER_CUBIC)

        for psm in psms:
            for wl in whitelists:
                text, data = _run_tesseract(img, _state.lang, psm, wl)
                conf = int(max(0, min(100, data.get("confidence", 0))))
                # pick longest alnum-heavy line
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pick = max(lines, key=lambda s: (_alpha_ratio(s), len(s))) if lines else text.strip()

                # scoring: primarily confidence, small tie-breakers
                score = conf + 0.75 * _alpha_ratio(pick) + 0.05 * min(len(pick), 40)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "text": pick,
                        "data": data,
                        "confidence": conf,
                        "tag": f"{tag}|psm{psm}|wl={'on' if wl else 'off'}",
                    }

    logger.debug("OCR best variant: %s", best["tag"])

    return {
        "text": best["text"].replace("\r", " ").strip(),
        "confidence": int(best["confidence"]),
        "boxes": best["data"].get("boxes", {}),
        "rotation": 0,                     # fixed (no orientation attempts)
        "roi_px": [x1, y1, x2, y2],
    }


# =========================
# Core steps
# =========================

def _preprocess(img_bgr: np.ndarray, st: OCRState) -> np.ndarray:
    """
    Prepare ROI for robust OCR. Returns a binarized or enhanced grayscale image.
    """
    img = img_bgr.copy()

    # Ensure sufficient width
    h, w = img.shape[:2]
    if w < st.min_ocr_width:
        scale = st.min_ocr_width / float(max(1, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Gamma (luma stretching)
    if st.gamma and abs(st.gamma - 1.0) > 1e-3:
        try:
            inv_gamma = 1.0 / max(1e-3, st.gamma)
            table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
            table = np.clip(table, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, table)
        except Exception:
            logger.debug("Gamma correction skipped", exc_info=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    if st.denoise:
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Small-angle deskew (not 180°)
    if st.enable_deskew:
        try:
            angle = _estimate_skew_angle(gray)
            if abs(angle) > 0.5:
                gray = _rotate_bound(gray, -angle)
        except Exception:
            logger.debug("Deskew skipped", exc_info=True)

    # Unsharp mask
    if st.unsharp:
        try:
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        except Exception:
            logger.debug("Unsharp skip", exc_info=True)

    # First attempt with adaptive; fallback to OTSU or raw gray
    bin_img = None
    if st.adaptive_thresh:
        try:
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
        except Exception:
            bin_img = None

    if bin_img is None and st.otsu_fallback:
        try:
            _thr, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            bin_img = gray

    # Optional inversion if background too bright
    if _state.invert_if_needed:
        try:
            if float(np.mean(bin_img)) > 180:
                bin_img = cv2.bitwise_not(bin_img)
        except Exception:
            pass

    return bin_img


def _run_tesseract(img: np.ndarray, lang: str, psm: int, whitelist: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Run tesseract with specified PSM and optional whitelist.
    Always uses LSTM (--oem 1) for best accuracy on Pi.
    """
    # Build config tokens safely (pytesseract will shlex.split this)
    tokens: List[str] = [f"--oem 1", f"--psm {int(psm)}"]
    if whitelist:
        tokens.append(f"-c tessedit_char_whitelist={_quote_tess_value(str(whitelist))}")
    config = " ".join(tokens)

    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)

        # Collect conf/boxes (word level)
        data: Dict[str, Any] = {}
        try:
            df = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)
            boxes = []
            confs = []
            n = len(df.get("text", []))
            for i in range(n):
                t = (df["text"][i] or "").strip()
                conf_raw = df["conf"][i]
                conf = int(conf_raw) if conf_raw not in ("-1", "", None) else -1
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
