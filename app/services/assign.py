# app/services/assign.py
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("assign")

# =========================
# State & configuration
# =========================

@dataclass
class AssignState:
    mode: str = "first_letter"          # "first_letter" | "csv_lookup"
    default_slot: Optional[int] = None  # used when no rule applies
    slots_csv: Optional[str] = None     # path for csv_lookup
    # Optional letter groups for first_letter mode:
    # e.g., {"ABC": 1, "DEF": 2, "GHIJK": 3}
    first_letter_groups: Dict[str, int] = field(default_factory=dict)

    # Runtime
    _csv_map: Dict[str, int] = field(default_factory=dict, repr=False)
    _csv_path: Optional[Path] = None
    _csv_mtime: Optional[float] = None


_state = AssignState()
_initialized = False

# Pattern to find first alphabetical character and normalize the title
_FIRST_ALPHA_RE = re.compile(r"[A-Za-z]")

# =========================
# Public API (used by main)
# =========================

def init(cfg: Optional[dict] = None) -> None:
    """
    Initialize assignment service.

    Config (all optional):
      mode: "first_letter" | "csv_lookup"
      default_slot: int or null
      slots_csv: "path/to/mapping.csv"
      first_letter_groups: { "ABC": 1, "DEF": 2, ... }  (only used in first_letter mode)
    """
    global _state, _initialized
    if cfg:
        _state.mode = cfg.get("mode", _state.mode)
        _state.default_slot = cfg.get("default_slot", _state.default_slot)
        _state.slots_csv = cfg.get("slots_csv", _state.slots_csv)
        flg = cfg.get("first_letter_groups")
        if isinstance(flg, dict):
            # normalize keys to uppercase, values to int
            norm = {}
            for letters, slot in flg.items():
                try:
                    norm[str(letters).upper()] = int(slot)
                except Exception:
                    logger.warning("Invalid first_letter_groups value: %r -> %r", letters, slot)
            _state.first_letter_groups = norm

    if _state.mode not in ("first_letter", "csv_lookup"):
        logger.warning("Unknown assignment mode '%s', falling back to 'first_letter'", _state.mode)
        _state.mode = "first_letter"

    # Prepare CSV if needed
    if _state.mode == "csv_lookup" and _state.slots_csv:
        _load_csv(force=True)

    _initialized = True
    logger.info(
        "Assign initialized (mode=%s, default_slot=%s, csv=%s, groups=%s)",
        _state.mode, _state.default_slot, _state.slots_csv, bool(_state.first_letter_groups)
    )


def decide(text: str) -> Dict[str, Optional[int] | str]:
    """
    Decide a slot for the given title text.

    Returns:
      { "slot_id": Optional[int], "rule_used": str }
    """
    if not _initialized:
        raise RuntimeError("Assignment service not initialized")

    title = (text or "").strip()
    if not title:
        return _result(None, "empty_title")

    if _state.mode == "csv_lookup":
        return _decide_csv(title)

    # default: first_letter
    return _decide_first_letter(title)


# =========================
# Mode: first_letter
# =========================

def _decide_first_letter(title: str) -> Dict[str, Optional[int] | str]:
    # Find the first alphabetical character in the string
    m = _FIRST_ALPHA_RE.search(title)
    if not m:
        # No alphabetical character found
        if _state.default_slot is not None:
            return _result(_state.default_slot, "default_slot_no_alpha")
        return _result(None, "no_alpha_char")

    first_char = title[m.start()].upper()

    # If groups are provided, use them first
    if _state.first_letter_groups:
        for group_letters, slot_id in _state.first_letter_groups.items():
            if first_char in group_letters:
                return _result(slot_id, f"first_letter_groups({first_char})")

    # Fallback: A=1 ... Z=26
    if "A" <= first_char <= "Z":
        slot = ord(first_char) - 64
        return _result(slot, "first_letter")

    # Non A–Z alpha (unlikely in English); apply default or none
    if _state.default_slot is not None:
        return _result(_state.default_slot, "default_slot_non_ascii_alpha")
    return _result(None, "non_ascii_alpha")


# =========================
# Mode: csv_lookup
# =========================

def _decide_csv(title: str) -> Dict[str, Optional[int] | str]:
    # Auto-reload CSV if it changed
    _load_csv(force=False)

    if not _state._csv_map:
        # No mapping loaded
        if _state.default_slot is not None:
            return _result(_state.default_slot, "default_slot_empty_csv")
        return _result(None, "csv_empty")

    key = _normalize_key(title)
    slot = _state._csv_map.get(key)
    if slot is not None:
        return _result(slot, "csv_lookup")

    # Not found → default?
    if _state.default_slot is not None:
        return _result(_state.default_slot, "default_slot_csv_miss")
    return _result(None, "csv_miss")


def _load_csv(force: bool) -> None:
    """
    Load or reload the CSV if needed.
    Expected columns (header names are case-insensitive):
      - 'key'     : string title key to match (after normalization)
      - 'slot_id' : integer slot id
    Any extra columns are ignored.
    """
    if not _state.slots_csv:
        return

    path = Path(_state.slots_csv)
    if not path.exists():
        logger.warning("slots_csv not found: %s", path)
        _state._csv_map.clear()
        _state._csv_path = None
        _state._csv_mtime = None
        return

    mtime = path.stat().st_mtime
    if not force and _state._csv_path == path and _state._csv_mtime == mtime:
        # Up-to-date
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize headers
            headers = {h.lower(): h for h in reader.fieldnames or []}
            key_col = headers.get("key")
            slot_col = headers.get("slot_id")
            if not key_col or not slot_col:
                raise ValueError("CSV must contain 'key' and 'slot_id' columns")

            mapping: Dict[str, int] = {}
            for row in reader:
                k_raw = (row.get(key_col) or "").strip()
                s_raw = (row.get(slot_col) or "").strip()
                if not k_raw or not s_raw:
                    continue
                try:
                    slot_id = int(s_raw)
                except Exception:
                    logger.warning("Invalid slot_id in CSV for key=%r: %r", k_raw, s_raw)
                    continue
                mapping[_normalize_key(k_raw)] = slot_id

        _state._csv_map = mapping
        _state._csv_path = path
        _state._csv_mtime = mtime
        logger.info("Loaded %d CSV assignments from %s", len(mapping), path)
    except Exception as e:
        logger.error("Failed to load slots_csv %s: %s", path, e, exc_info=True)
        # Clear mapping on failure to avoid stale state
        _state._csv_map.clear()
        _state._csv_path = path
        _state._csv_mtime = mtime


# =========================
# Helpers
# =========================

def _normalize_key(s: str) -> str:
    """
    Normalization for CSV keys and lookup:
      - strip leading/trailing whitespace
      - collapse internal whitespace to single spaces
      - uppercase
    """
    s = " ".join((s or "").split())
    return s.upper()


def _result(slot_id: Optional[int], rule: str) -> Dict[str, Optional[int] | str]:
    return {"slot_id": slot_id, "rule_used": rule}
