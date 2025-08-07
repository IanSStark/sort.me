# sorter.py
"""
Translate the first letter of the card title into a physical bin (zone) and
return both the zone name and its (X, Y) coordinate on the MPCNC bed.
"""

from typing import List, Dict, Tuple, Optional

# ----- DEFINE YOUR BINS BELOW -----
ZONES: List[Dict[str, object]] = [
    {"name": "A-D", "letters": "ABCD", "coord": (10.0,  50.0)},
    {"name": "E-H", "letters": "EFGH", "coord": (30.0,  50.0)},
    {"name": "I-L", "letters": "IJKL", "coord": (50.0,  50.0)},
    {"name": "M-P", "letters": "MNOP", "coord": (70.0,  50.0)},
    {"name": "Q-T", "letters": "QRST", "coord": (90.0,  50.0)},
    {"name": "U-Z", "letters": "UVWXYZ", "coord": (110.0, 50.0)},
]


def determine_zone(title: str) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    if not title:
        return None, None
    first = title[0].upper()
    for zone in ZONES:
        if first in zone["letters"]:
            return zone["name"], zone["coord"]  # type: ignore
    return None, None


if __name__ == "__main__":
    for test in ("Lightning", "Counterspell", "Zephyr"):
        print(test, "→", determine_zone(test))
