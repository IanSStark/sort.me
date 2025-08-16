# app/services/ocr.py
from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

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
    psm: int = 6                 # Block of text (TL->BR)
    oem: int = 1                 # LSTM engine
    whitelist: Optional[str] = None

    # Selection / acceptance policy
    min_accept_confidence: int = 75     # raised baseline acceptance threshold
    top_priority_fraction: float = 0.35 # top X% of the image gets priority
    top_bias_bonus: float = 10.0        # score bonus if line center is in top region
    min_line_chars: int = 3             # ignore ultra-short “lines”

    initialized: bool = False

_state = OCRState()

# =========================
# Public API (expected by main.py)
# =========================

def init(cfg: Dict[str, Any]) -> None:
    """
    Initialize OCR. We OCR the *entire image* with no rotation/cropping, but
    selection prioritizes top-of-image lines and enforces a higher acceptance threshold.

    Supported cfg keys (under `ocr:`):
      engine, lang, psm, oem, whitelist
      min_accept_confidence, top_priority_fraction, top_bias_bonus, min_line_chars
    """
    global _state
    _state.engine = str(cfg.get("engine", _state.engine))
    _state.lang = str(cfg.get("lang", _state.lang))
    _state.psm = int(cfg.get("psm", _state.psm))
    _state.oem = int(cfg.get("oem", _state.oem))
    wl = cfg.get("whitelist", _state.whitelist)
    _state.whitelist = None if wl in (None, "", "null", "None") else str(wl)

    _state.min_accept_confidence = int(cfg.get("min_accept_confidence", _state.min_accept_confidence))
    _state.top_priority_fraction = float(cfg.get("top_priority_fraction", _state.top_priority_fraction))
    _state.top_priority_fraction = max(0.05, min(0.75, _state.top_priority_fraction))  # guardrails
    _state.top_bias_bonus = float(cfg.get("top_bias_bonus", _state.top_bias_bonus))
    _state.min_line_chars = int(cfg.get("min_line_chars", _state.min_line_chars))

    if _state.engine.lower() == "tesseract":
        try:
            out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=False)
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or "tesseract not available")
        except FileNotFoundError:
            raise RuntimeError("tesseract binary not found; install tesseract-ocr")

    _state.initialized = True
    logger.info(
        "OCR initialized: engine=%s lang=%s psm=%d oem=%d min_accept_conf=%d top_fraction=%.2f top_bonus=%.1f",
        _state.engine, _state.lang, _state.psm, _state.oem,
        _state.min_accept_confidence, _state.top_priority_fraction, _state.top_bias_bonus
    )

def status() -> bool:
    return _state.initialized

def run(image_path: str) -> Dict[str, Any]:
    """
    Perform OCR on the full image (no rotation/cropping), reading TL->BR.
    Then select the *best line*, prioritizing lines near the *top* of the image,
    and enforce an acceptance confidence threshold.

    Returns:
      {
        'text': str,
        'confidence': int,            # selected line avg confidence (0..100)
        'boxes': {'words': [...]},    # all words & per-word conf/bboxes
        'rotation': 0,                # fixed
        'roi_px': [0,0,w,h],          # full image
        'accepted': bool,             # True if confidence >= min_accept_confidence
        'selection': {                # metadata about the chosen line
            'line_bbox': [l,t,w,h],
            'line_conf': int,
            'line_center_y': int,
            'in_top_region': bool,
            'score': float
        }
      }
    """
    if not _state.initialized:
        raise RuntimeError("OCR not initialized")

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    H, W = img_bgr.shape[:2]

    # Build Tesseract config
    config = f"--oem {_state.oem} --psm {_state.psm}"
    if _state.whitelist:
        wl = _state.whitelist.replace('\\', '\\\\').replace('"', '\\"')
        config += f' -c tessedit_char_whitelist="{wl}"'

    # Full-text (optionally useful)
    try:
        full_text = pytesseract.image_to_string(img_bgr, lang=_state.lang, config=config)
    except Exception as e:
        logger.error("image_to_string failed: %s", e, exc_info=True)
        full_text = ""

    # Word-level data (we'll reconstruct lines)
    boxes: Dict[str, Any] = {"words": []}
    line_map: DefaultDict[Tuple[int, int, int], List[int]] = defaultdict(list)
    # key: (block_num, par_num, line_num) -> list of word indices

    try:
        df = pytesseract.image_to_data(img_bgr, lang=_state.lang, config=config, output_type=Output.DICT)
        n = len(df.get("text", []))
        for i in range(n):
            t = (df["text"][i] or "").strip()
            conf_raw = df["conf"][i]
            conf = int(conf_raw) if conf_raw not in ("-1", "", None) else -1

            left = int(df["left"][i]); top = int(df["top"][i])
            width = int(df["width"][i]); height = int(df["height"][i])

            # Keep all words (even blanks help with geometry sometimes)
            boxes["words"].append({
                "text": t,
                "conf": conf,
                "left": left, "top": top, "width": width, "height": height,
                "block": int(df.get("block_num", [0])[i]),
                "par": int(df.get("par_num", [0])[i]),
                "line": int(df.get("line_num", [0])[i]),
                "word": int(df.get("word_num", [0])[i]),
            })

            if t:  # only index non-empty words into lines
                key = (
                    int(df.get("block_num", [0])[i]),
                    int(df.get("par_num", [0])[i]),
                    int(df.get("line_num", [0])[i]),
                )
                line_map[key].append(i)

    except Exception as e:
        logger.error("image_to_data failed: %s", e, exc_info=True)
        # If this fails, fall back to the raw full_text with heuristic confidence
        est_conf = _estimate_confidence(full_text)
        return {
            "text": (full_text or "").replace("\r", " ").strip(),
            "confidence": int(est_conf),
            "boxes": boxes,
            "rotation": 0,
            "roi_px": [0, 0, W, H],
            "accepted": bool(est_conf >= _state.min_accept_confidence),
            "selection": {
                "line_bbox": [0, 0, W, H],
                "line_conf": int(est_conf),
                "line_center_y": H // 6,
                "in_top_region": True,
                "score": float(est_conf),
            },
        }

    # Build line candidates with geometry and average confidence
    top_cut = int(round(H * _state.top_priority_fraction))
    best = {
        "score": -1e9,
        "text": "",
        "avg_conf": 0,
        "bbox": (0, 0, W, max(1, H // 6)),
        "center_y": max(1, H // 6),
        "in_top": True,
    }

    for key, idxs in line_map.items():
        # Gather words in order of x (left to right) to produce natural reading
        idxs_sorted = sorted(idxs, key=lambda i: boxes["words"][i]["left"])
        words = [boxes["words"][i] for i in idxs_sorted]
        line_text = " ".join(w["text"] for w in words if w["text"])
        if len(line_text.strip()) < _state.min_line_chars:
            continue

        # Line bbox
        l = min(w["left"] for w in words)
        t = min(w["top"] for w in words)
        r = max((w["left"] + w["width"]) for w in words)
        b = max((w["top"] + w["height"]) for w in words)
        bbox = (l, t, max(1, r - l), max(1, b - t))
        center_y = t + bbox[3] // 2

        # Confidence aggregation
        confs = [w["conf"] for w in words if isinstance(w["conf"], int) and w["conf"] >= 0]
        avg_conf = int(round(float(np.mean(confs)))) if confs else _estimate_confidence(line_text)

        # Scoring: confidence + top-region bias + small length bonus
        in_top = center_y <= top_cut
        score = avg_conf + (_state.top_bias_bonus if in_top else 0.0) + 0.05 * min(len(line_text), 40)

        if (score > best["score"]) or (score == best["score"] and avg_conf > best["avg_conf"]):
            best = {
                "score": score,
                "text": line_text.strip(),
                "avg_conf": int(avg_conf),
                "bbox": bbox,
                "center_y": int(center_y),
                "in_top": bool(in_top),
            }

    # Fallback if no line candidates (e.g., image had no words)
    if best["text"] == "":
        est_conf = _estimate_confidence(full_text)
        return {
            "text": (full_text or "").replace("\r", " ").strip(),
            "confidence": int(est_conf),
            "boxes": boxes,
            "rotation": 0,
            "roi_px": [0, 0, W, H],
            "accepted": bool(est_conf >= _state.min_accept_confidence),
            "selection": {
                "line_bbox": [0, 0, W, H],
                "line_conf": int(est_conf),
                "line_center_y": H // 6,
                "in_top_region": True,
                "score": float(est_conf),
            },
        }

    # Normal case: return best line text + confidence
    accepted = best["avg_conf"] >= _state.min_accept_confidence
    return {
        "text": best["text"],
        "confidence": int(best["avg_conf"]),
        "boxes": boxes,           # all words remain available for debugging
        "rotation": 0,
        "roi_px": [0, 0, W, H],   # full image region scanned
        "accepted": bool(accepted),
        "selection": {
            "line_bbox": [int(best["bbox"][0]), int(best["bbox"][1]), int(best["bbox"][2]), int(best["bbox"][3])],
            "line_conf": int(best["avg_conf"]),
            "line_center_y": int(best["center_y"]),
            "in_top_region": bool(best["in_top"]),
            "score": float(best["score"]),
        },
    }

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
