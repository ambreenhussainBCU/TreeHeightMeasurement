"""
Machine vision checks used by the app (no Kivy code here).

check_readings()        After you add a blue dot: draw the ruler onto the picture
                        exactly as Save would, then let machine vision read every
                        banner back and measure every dot. Each reading gets a
                        verdict such as "Vision 11.53" (confirmed) or a warning.

find_existing_ruler()   When you open a picture: does it already contain a Tree
                        Divider ruler (e.g. a picture saved earlier)? If so, return
                        its base, top and readings so the app can load them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import exporter
from .ruler import Point

MAX_SIDE = 1600          # vision works on a copy at most this big: fast, and plenty sharp
MATCH = 0.01             # a detected dot belongs to a reading if within 1% of tree height


def available() -> bool:
    try:
        from . import vision  # noqa: F401  (needs OpenCV + numpy)
        return True
    except ImportError:
        return False


@dataclass
class Check:
    text: str
    status: str          # "ok" | "check" | "measured" | "not_found" | "pending"


PENDING = Check("Vision: reading...", "pending")


def _shrink(picture, max_side=MAX_SIDE):
    scale = min(1.0, max_side / max(picture.size))
    if scale < 1.0:
        picture = picture.resize((round(picture.width * scale), round(picture.height * scale)))
    return picture, scale


def check_readings(picture, base: Point, top: Point, parts: int,
                   fractions: List[float], use_ocr: bool = True) -> List[Check]:
    """Verdict for each reading (same order as `fractions`)."""
    from . import vision
    small, k = _shrink(picture)
    marked = exporter.render(small, Point(base.x * k, base.y * k), Point(top.x * k, top.y * k),
                             parts, fractions)
    try:
        res = vision.analyse(vision.pil_to_bgr(marked), parts, use_ocr=use_ocr)
    except ValueError:
        return [Check("Vision: ruler not found", "not_found") for _ in fractions]
    out = []
    for f in fractions:
        expected = round(f * parts, 2)
        if not res.fractions:
            out.append(Check("Vision: dot not found", "not_found"))
            continue
        j = min(range(len(res.fractions)), key=lambda i: abs(res.fractions[i] - f))
        if abs(res.fractions[j] - f) > MATCH:
            out.append(Check("Vision: dot not found", "not_found"))
            continue
        r = res.readings[j]
        if r.ocr is not None and r.agree and abs(r.ocr - expected) < 0.006:
            out.append(Check(f"Vision {r.ocr:.2f}", "ok"))
        elif r.ocr is not None:
            out.append(Check(f"Vision: banner {r.ocr:.2f}, measured {r.geometric:.2f}", "check"))
        else:
            out.append(Check(f"Vision {r.geometric:.2f} (measured)", "measured"))
    return out


@dataclass
class FoundRuler:
    base: Point
    top: Point
    parts: int
    fractions: List[float]
    values: List[float]          # what the banners say (or the measurement)
    parts_guessed: bool          # True if `parts` was worked out from the banner text


def find_existing_ruler(picture, parts: int) -> Optional[FoundRuler]:
    """Look for a ruler already drawn in `picture` (full-size Pillow image)."""
    from . import vision
    small, k = _shrink(picture)
    try:
        res = vision.analyse(vision.pil_to_bgr(small), parts)
    except ValueError:
        return None
    if not res.readings:
        return None             # a yellow/pink stripe alone isn't convincing enough

    # The banner says the value and the measurement says the fraction, so
    # value / fraction = number of parts the picture was saved with.
    guesses = [r.ocr / f for r, f in zip(res.readings, res.fractions)
               if r.ocr is not None and f > 0.05]
    parts_guessed = False
    if guesses:
        g = round(sum(guesses) / len(guesses))
        if 2 <= g <= 200 and all(abs(x - g) <= max(1.0, g * 0.02) for x in guesses):
            parts, parts_guessed = g, True

    values, fractions = [], []
    for r, f in zip(res.readings, res.fractions):
        geo = round(f * parts, 2)
        if r.ocr is not None and abs(r.ocr - geo) <= max(0.05, parts * 0.01):
            # Banner text confirmed: place the dot exactly at the printed value
            # (the pixel measurement is only accurate to about 0.05% of the height).
            values.append(r.ocr)
            fractions.append(r.ocr / parts)
        else:
            values.append(geo)
            fractions.append(f)
    return FoundRuler(Point(res.base_xy[0] / k, res.base_xy[1] / k),
                      Point(res.top_xy[0] / k, res.top_xy[1] / k),
                      parts, fractions, values, parts_guessed)
