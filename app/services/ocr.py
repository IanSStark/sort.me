# app/services/ocr.py
from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    psm: int = 6                 # block of text, left->right, top->bottom
    oem: int = 1                 # LSTM (1) for accuracy; 3=default; both fine on Pi
    whitelist: Optional[str] = None
    initialized: bool = False

_state = OCRState()

# =========================
# Public API (expected by main.py)
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize OCR. We OCR the *entire image* with no rotation/cropping.
    Supported cfg keys: engine, lang, psm, oem, whitelist
    """
    global _state
    _state.engine = str(cfg.get("engine", _state.engine))
    _state.lang = str(cfg.get("lang", _state.lang))
    _state.psm = int(cfg.get("psm", _state.psm))
    _state.oem = int(cfg.get("oem", _state.oem))
    wl = cfg.get("whitelist", _state.whitelist)
    _state.whitelist = None if wl in (None, "", "null", "None") else str(wl)

    if _state.engine.lower() == "tesseract":
        try:
            out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=False)
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or "tesseract not available")
        except FileNotFoundError:
            raise RuntimeError("tesseract binary not found; install tesseract-ocr")

    _state.initialized = True
    logger.info("OCR initialized: engine=%s lang=%s psm=%d oem=%d", _state.engine, _state.lang, _state.psm, _state.oem)

def status() -> bool:
    return _state.initialized

def run(image_path: str) -> Dict[str, Any]:
    """
    Perform OCR on the full image (no rotation/cropping), reading TL->BR.
    Returns:
      {
        'text': str,
        'confidence': int,          # 0..100 (aggregated)
        'boxes': {'words': [...]},  # word boxes & per-word conf
        'rotation': 0,              # fixed
        'roi_px': [0,0,w,h]         # whole image
      }
    """
    if not _state.initialized:
        raise RuntimeError("OCR not initialized")

    # Read with OpenCV; robust to non-ASCII paths
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img_bgr.shape[:2]

    config = f"--oem {_state.oem} --psm {_state.psm}"
    if _state.whitelist:
        # Safe quoting for pytesseract (avoid shlex issues)
        wl = _state.whitelist.replace('\\', '\\\\').replace('"', '\\"')
        config += f' -c tessedit_char_whitelist="{wl}"'

    try:
        # Text (full image)
        text = pytesseract.image_to_string(img_bgr, lang=_state.lang, config=config)

        # Word-level data
        boxes: Dict[str, Any] = {"words": []}
        confs: List[int] = []
        try:
            df = pytesseract.image_to_data(img_bgr, lang=_state.lang, config=config, output_type=Output.DICT)
            n = len(df.get("text", []))
            for i in range(n):
                t = (df["text"][i] or "").strip()
                conf_raw = df["conf"][i]
                conf = int(conf_raw) if conf_raw not in ("-1", "", None) else -1
                if t:
                    boxes["words"].append({
                        "text": t,
                        "conf": conf,
                        "left": int(df["left"][i]),
                        "top": int(df["top"][i]),
                        "width": int(df["width"][i]),
                        "height": int(df["height"][i]),
                    })
                confs.append(conf)
            agg_conf = _aggregate_confidence(confs, text)
        except Exception:
            boxes = {"words": []}
            agg_conf = _estimate_confidence(text)

        return {
            "text": (text or "").replace("\r", " ").strip(),
            "confidence": int(max(0, min(100, agg_conf))),
            "boxes": boxes,
            "rotation": 0,
            "roi_px": [0, 0, w, h],
        }

    except Exception as e:
        logger.error("OCR failed: %s", e, exc_info=True)
        raise RuntimeError("OCR text extraction failed")

# =========================
# Confidence helpers
# =========================

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
