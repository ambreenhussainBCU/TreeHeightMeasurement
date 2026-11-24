"""
Geometry for the tree ruler.

This module has no Kivy or Pillow imports, so the same maths drives the
on-screen overlay, the saved image, and the unit tests.

Coordinates can be in any system (screen pixels with y up, or image pixels
with y down); the ruler only cares about the base point and the top point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

# Allowed gaps between printed numbers, smallest first.
LABEL_STEPS = (1, 2, 5, 10, 20, 25, 50, 100)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, o: "Point") -> "Point":
        return Point(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Point") -> "Point":
        return Point(self.x - o.x, self.y - o.y)

    def scale(self, k: float) -> "Point":
        return Point(self.x * k, self.y * k)

    def dot(self, o: "Point") -> float:
        return self.x * o.x + self.y * o.y

    def dist(self, o: "Point") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)


@dataclass(frozen=True)
class Tick:
    index: int      # 0 = base, parts = top
    start: Point
    end: Point
    kind: str       # "major" (every 10), "mid" (every 5) or "minor"


@dataclass(frozen=True)
class Band:
    start: Point
    end: Point
    alternate: bool  # False = yellow, True = pink


@dataclass(frozen=True)
class Label:
    value: int
    anchor: Point    # centre of the label
    major: bool      # multiple of 10
    centred: bool    # True = sits in the middle of a part, False = marks a boundary


class Ruler:
    """A straight ruler from `base` (0) to `top` (`parts`)."""

    def __init__(self, base: Point, top: Point, parts: int = 100):
        if parts < 1:
            raise ValueError("parts must be at least 1")
        self.base = base
        self.top = top
        self.parts = parts
        self.vector = top - base
        self.length = math.hypot(self.vector.x, self.vector.y)
        if self.length == 0:
            raise ValueError("base and top must be different points")
        self.unit = self.vector.scale(1 / self.length)          # along the trunk
        self.normal = Point(-self.unit.y, self.unit.x)          # across the trunk
        # Always put the numbers on the right-hand side of the screen/image.
        self.side = 1 if (self.normal.x > 0 or (self.normal.x == 0 and self.normal.y > 0)) else -1

    # ---- basic measurements ------------------------------------------------
    @property
    def part_length(self) -> float:
        """Length of one part in the same units as the points."""
        return self.length / self.parts

    def point_at(self, fraction: float) -> Point:
        """Point on the ruler at 0.0 (base) .. 1.0 (top)."""
        return self.base + self.vector.scale(fraction)

    def fraction_of(self, p: Point, clamp: bool = True) -> float:
        """Project any point onto the ruler and return its position 0..1."""
        t = (p - self.base).dot(self.vector) / (self.length ** 2)
        return min(1.0, max(0.0, t)) if clamp else t

    def reading(self, p: Point) -> float:
        """The ruler reading (0..parts) for a tapped point, e.g. 11.64."""
        return self.fraction_of(p) * self.parts

    def distance_to_line(self, p: Point) -> float:
        """How far a point is from the ruler line, sideways."""
        return abs((p - self.base).dot(self.normal))

    def beside(self, fraction: float, offset: float) -> Point:
        """A point `offset` away from the ruler on the number side."""
        return self.point_at(fraction) + self.normal.scale(self.side * offset)

    # ---- drawing data ------------------------------------------------------
    def bands(self) -> List[Band]:
        """Alternating colour bands: every part if few, every 10 if many."""
        group = 10 if self.parts > 20 else 1
        out = []
        for i in range(0, self.parts, group):
            a = i / self.parts
            b = min(i + group, self.parts) / self.parts
            out.append(Band(self.point_at(a), self.point_at(b), (i // group) % 2 == 1))
        return out

    def ticks(self, major: float = 34, mid: float = 24, minor: float = 16) -> List[Tick]:
        """A tick at every division, longer at every 5th and 10th."""
        out = []
        for i in range(self.parts + 1):
            kind = "major" if i % 10 == 0 else "mid" if i % 5 == 0 else "minor"
            half = {"major": major, "mid": mid, "minor": minor}[kind] / 2
            c = self.point_at(i / self.parts)
            out.append(Tick(i, c - self.normal.scale(half), c + self.normal.scale(half), kind))
        return out

    def label_step(self, min_spacing: float) -> int:
        """Smallest step that keeps printed numbers at least `min_spacing` apart."""
        for step in LABEL_STEPS:
            if step * self.part_length >= min_spacing:
                return step
        return self.parts

    def labels(self, min_spacing: float, offset: float) -> List[Label]:
        """Numbers to print beside the ruler, thinned out so they never overlap."""
        step = self.label_step(min_spacing)
        if step == 1 and self.parts <= 20:
            # Few parts: number each part in its middle (1..parts).
            return [
                Label(i + 1, self.beside((i + 0.5) / self.parts, offset), (i + 1) % 10 == 0, True)
                for i in range(self.parts)
            ]
        # Many parts: tape-measure style, a number marks a position (11 = 11 parts up),
        # so a reading of 11.64 sits between the 11 and 12 marks at any zoom.
        return [
            Label(v, self.beside(v / self.parts, offset), v % 10 == 0, False)
            for v in range(step, self.parts + 1, step)
        ]


# ---- mapping between the screen and the picture ----------------------------
@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def fit_contain(box: Rect, image_w: float, image_h: float) -> Rect:
    """Where an image appears when letter-boxed ("contain") inside `box`."""
    scale = min(box.w / image_w, box.h / image_h)
    w, h = image_w * scale, image_h * scale
    return Rect(box.x + (box.w - w) / 2, box.y + (box.h - h) / 2, w, h)


def screen_to_image(p: Point, shown: Rect, image_w: float, image_h: float,
                    screen_y_up: bool = True) -> Point:
    """Convert a screen point to a pixel position in the full-size picture.

    Kivy's screen origin is bottom-left (y up); image pixels start top-left.
    """
    fx = (p.x - shown.x) / shown.w
    fy = (p.y - shown.y) / shown.h
    if screen_y_up:
        fy = 1 - fy
    return Point(fx * image_w, fy * image_h)
