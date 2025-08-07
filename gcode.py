# gcode.py
"""
Produce a minimal Marlin-compatible G-code fragment that moves over the bin,
lowers the gripper, drops the card, and retracts.
"""

from typing import Tuple, List


def generate_gcode(
    coord: Tuple[float, float],
    feed_xy: int = 600,   # mm/min
    feed_z: int = 300,
    safe_z: float = 10.0,  # clearance above cards
    place_z: float = -2.0  # height to drop card
) -> str:
    x, y = coord
    lines: List[str] = [
        "; === Start card move ===",
        "G90",                                   # absolute positioning
        f"G01 Z{safe_z:.3f} F{feed_z}",          # raise to safe height
        f"G01 X{x:.3f} Y{y:.3f} F{feed_xy}",     # travel over bin
        f"G01 Z{place_z:.3f} F{feed_z}",         # lower to placement depth
        "; (activate gripper OFF here, if required)",
        f"G01 Z{safe_z:.3f} F{feed_z}",          # lift up again
        "G01 X0.000 Y0.000 F600",                # return to origin
        "; === End card move ===",
    ]
    return "\n".join(lines) + "\n"
