"""
The app's state and what each tap does. No UI code here.

Flow:  tap 1 -> base,  tap 2 -> top,  later taps near the trunk -> readings.
Points are stored in picture pixels. Readings are stored as fractions (0..1)
so they stay on the same spot of the tree when you drag the base/top or change
the number of parts.

`scale` (screen pixels per picture pixel) keeps finger-sized hit areas the same
size on screen at any zoom level.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

from . import height
from .ruler import Point, Ruler

Target = Union[str, Tuple[str, int]]   # "base", "top" or ("reading", index)

MIN_PARTS, MAX_PARTS = 2, 200


class Session:
    def __init__(self, parts: int = 100, hit_radius: float = 36, read_radius: float = 80):
        self.parts = parts
        self.hit_radius = hit_radius      # how close a tap must be to grab a dot
        self.read_radius = read_radius    # how close to the trunk a tap must be to read
        self.base: Optional[Point] = None
        self.top: Optional[Point] = None
        self.readings: List[float] = []
        self.dragging: Optional[Target] = None
        self.scale = 1.0
        self.banner_m = height.DEFAULT_BANNER_M   # height of the reference banner, in metres

    # ---- derived state -----------------------------------------------------
    @property
    def stage(self) -> str:
        if self.base is None:
            return "base"
        if self.top is None:
            return "top"
        return "ready"

    @property
    def ruler(self) -> Optional[Ruler]:
        if self.base is None or self.top is None or self.base == self.top:
            return None
        return Ruler(self.base, self.top, self.parts)

    def reading_value(self, index: int) -> float:
        return self.readings[index] * self.parts

    def hint(self) -> str:
        if self.stage == "ready" and not self.readings:
            return f"Now tap the top of the {self.banner_m:g} m banner"
        return {
            "base": "Tap the base of the tree",
            "top": "Now tap the top of the tree",
            "ready": "Tap to add more points  ·  double-tap to remove",
        }[self.stage]

    def heights(self) -> Optional[height.Heights]:
        """Tree height from the first reading (the banner's top), or None."""
        if self.ruler is None:
            return None
        return height.compute([f * self.parts for f in self.readings], self.parts, self.banner_m)

    def height_labels(self) -> List[str]:
        """Small note under each banner: the banner itself, then heights of other points."""
        h = self.heights()
        if h is None:
            return []
        return [f"Banner {h.banner_m:g} m"] + [f"{m:.2f} m above base" for _, m in h.points]

    # ---- hit testing -------------------------------------------------------
    def target_at(self, p: Point) -> Optional[Target]:
        candidates = []
        ruler = self.ruler
        if ruler:
            for i, f in enumerate(self.readings):
                candidates.append((ruler.point_at(f).dist(p), ("reading", i)))
        if self.top is not None:
            candidates.append((self.top.dist(p), "top"))
        if self.base is not None:
            candidates.append((self.base.dist(p), "base"))
        radius = self.hit_radius / self.scale
        hits = [c for c in candidates if c[0] <= radius]
        return min(hits, key=lambda c: c[0])[1] if hits else None

    # ---- touch handling ----------------------------------------------------
    def grab(self, p: Point) -> Optional[Target]:
        """Start dragging whatever dot is under `p` (if any)."""
        self.dragging = self.target_at(p)
        return self.dragging

    def tap(self, p: Point) -> Optional[str]:
        """A tap on empty space: set base, then top, then add readings."""
        if self.base is None:
            self.base = p
            return "base_set"
        if self.top is None:
            if p.dist(self.base) * self.scale < 1:
                return None
            self.top = p
            return "top_set"
        ruler = self.ruler
        if ruler and ruler.distance_to_line(p) * self.scale <= self.read_radius:
            t = ruler.fraction_of(p, clamp=False)
            if 0 <= t <= 1:
                self.readings.append(t)
                return "reading_added"
        return None

    def delete_reading(self, index: int) -> None:
        del self.readings[index]

    def press(self, p: Point, double_tap: bool = False) -> Optional[str]:
        """Grab a dot, delete a double-tapped reading, or tap. (Convenience for simple UIs.)"""
        target = self.target_at(p)
        if target is not None:
            if double_tap and isinstance(target, tuple):
                self.delete_reading(target[1])
                return "reading_removed"
            self.dragging = target
            return "grabbed"
        event = self.tap(p)
        if event == "reading_added":
            self.dragging = ("reading", len(self.readings) - 1)
        return event

    def drag(self, p: Point) -> None:
        if self.dragging == "base":
            if self.top is None or p != self.top:
                self.base = p
        elif self.dragging == "top":
            if p != self.base:
                self.top = p
        elif isinstance(self.dragging, tuple):
            ruler = self.ruler
            if ruler:
                self.readings[self.dragging[1]] = ruler.fraction_of(p)

    def release(self) -> None:
        self.dragging = None

    # ---- buttons -----------------------------------------------------------
    def set_parts(self, parts: int) -> None:
        self.parts = max(MIN_PARTS, min(MAX_PARTS, int(parts)))

    def step_parts(self, direction: int) -> None:
        """-/+ buttons: steps of 1 up to 20, then steps of 10."""
        if direction > 0:
            step = 10 if self.parts >= 20 else 1
        else:
            step = 10 if self.parts > 20 else 1
        self.set_parts(self.parts + direction * step)

    def clear_readings(self) -> None:
        self.readings.clear()

    def reset(self) -> None:
        self.base = self.top = None
        self.readings.clear()
        self.dragging = None
