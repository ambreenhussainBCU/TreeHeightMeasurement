"""
Tree height in metres, using a banner (reference object) of known height.

Put the banner upright at the base of the tree, mark the BANNER'S TOP with the first
blue dot, and the ruler tells you what share of the tree the banner is:

    banner reading 11.53 (of 100)  ->  the banner is 11.53 % of the tree
    tree height = banner height x parts / banner reading
                = 1 m x 100 / 11.53 = 8.67 m

Any other blue dot gives the height of that point above the base:

    point height = point reading / banner reading x banner height

Also builds the lines written to the readings.txt log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

DEFAULT_BANNER_M = 1.0
LOG_NAME = "readings.txt"
LOG_COLUMNS = ["Date", "Time", "Picture", "Parts", "Banner reading", "Banner height (m)",
               "Tree height (m)", "Other readings (reading = height above base)"]


@dataclass
class Heights:
    parts: int
    banner_m: float
    banner_reading: float
    tree_m: float
    points: List[tuple] = field(default_factory=list)   # (reading, metres) for other dots

    def point_text(self, reading: float, metres: float) -> str:
        return f"{reading:.2f} = {metres:.2f} m"


def tree_height(banner_reading: float, parts: int, banner_m: float = DEFAULT_BANNER_M) -> float:
    if banner_reading <= 0:
        raise ValueError("the banner reading must be above the base (greater than 0)")
    if banner_m <= 0:
        raise ValueError("the banner height must be greater than 0")
    return banner_m * parts / banner_reading


def point_height(reading: float, banner_reading: float, banner_m: float = DEFAULT_BANNER_M) -> float:
    return reading / banner_reading * banner_m


def compute(readings: Sequence[float], parts: int,
            banner_m: float = DEFAULT_BANNER_M) -> Optional[Heights]:
    """`readings` in order; the first one is the banner's top. None if there's nothing usable."""
    if not readings or readings[0] <= 0 or banner_m <= 0:
        return None
    banner = readings[0]
    return Heights(parts, banner_m, banner, tree_height(banner, parts, banner_m),
                   [(r, point_height(r, banner, banner_m)) for r in readings[1:]])


# ---- text log ---------------------------------------------------------------------------
def log_line(picture: str, h: Optional[Heights], readings: Sequence[float], parts: int,
             when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    if h is None:
        banner = banner_m = tree = ""
        others = ", ".join(f"{r:.2f}" for r in readings)
    else:
        banner, banner_m, tree = f"{h.banner_reading:.2f}", f"{h.banner_m:.2f}", f"{h.tree_m:.2f}"
        others = ", ".join(h.point_text(r, m) for r, m in h.points)
    cells = [when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S"), picture, str(parts),
             banner, banner_m, tree, others]
    return "\t".join(cells)


def append_log(path: str, line: str) -> None:
    """Add one line to the log, creating it (with a header) if needed."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write("Tree Divider readings (tab-separated: opens in Excel or any text editor)\n")
            f.write("\t".join(LOG_COLUMNS) + "\n")
        f.write(line + "\n")
