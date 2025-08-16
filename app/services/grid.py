# app/services/grid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Zone:
    key: str
    type: str      # 'feeder' | 'sort' | ...
    row: int
    col: int
    x_mm: float
    y_mm: float

class Grid:
    def __init__(self, cfg: Dict[str, Any]):
        self.rows = int(cfg.get("rows", 1))
        self.cols = int(cfg.get("cols", 1))
        ox, oy = cfg.get("origin_xy_mm", [0.0, 0.0])
        px, py = cfg.get("pitch_xy_mm", [0.0, 0.0])
        self.origin_x = float(ox); self.origin_y = float(oy)
        self.pitch_x  = float(px); self.pitch_y  = float(py)
        self.row_major = bool(cfg.get("row_major", True))
        self.overrides: Dict[str, List[float]] = cfg.get("overrides", {}) or {}
        self.alpha_map: Dict[str, str] = cfg.get("alpha_map", {}) or {}

        zones_cfg = cfg.get("zones", []) or []
        zones: Dict[str, Zone] = {}
        for z in zones_cfg:
            key = str(z["key"]).strip()
            ztype = str(z.get("type", "sort")).strip()
            r = int(z["row"]); c = int(z["col"])
            x, y = self.slot_rc_to_xy(r, c)
            if key in self.overrides:
                ox, oy = self.overrides[key]
                x += float(ox); y += float(oy)
            zones[key] = Zone(key=key, type=ztype, row=r, col=c, x_mm=x, y_mm=y)
        self._zones = zones

    def slot_rc_to_xy(self, row: int, col: int) -> Tuple[float, float]:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"slot ({row},{col}) out of bounds for {self.rows}x{self.cols}")
        x = self.origin_x + col * self.pitch_x
        y = self.origin_y + row * self.pitch_y
        return x, y

    def get_zone(self, key: str) -> Zone:
        z = self._zones.get(key)
        if not z:
            raise KeyError(f"zone '{key}' not defined")
        return z

    def list_zones(self) -> List[Dict[str, Any]]:
        return [dict(key=z.key, type=z.type, row=z.row, col=z.col, x_mm=z.x_mm, y_mm=z.y_mm)
                for z in self._zones.values()]

    def letter_to_zone(self, letter: str) -> Optional[Zone]:
        if not letter:
            return None
        key = self.alpha_map.get(letter.upper()[0])
        return self._zones.get(key) if key else None

    def slotid_to_rc(self, slot_id: int) -> Tuple[int, int]:
        """
        Convert 1-based slot id (A=1..Z=26 style) into (row,col) using row_major numbering.
        """
        if slot_id < 1 or slot_id > self.rows * self.cols:
            raise ValueError(f"slot_id {slot_id} out of bounds for {self.rows*self.cols} slots")
        idx0 = slot_id - 1
        if self.row_major:
            row = idx0 // self.cols
            col = idx0 % self.cols
        else:
            col = idx0 // self.rows
            row = idx0 % self.rows
        return row, col
