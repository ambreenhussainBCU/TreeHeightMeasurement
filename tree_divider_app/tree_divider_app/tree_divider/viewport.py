"""
Zoom and pan: converts between picture pixels and screen positions.

Picture coordinates: origin at the top-left of the full-size photo, y down.
Screen coordinates:  Kivy style, origin bottom-left, y up.

The base, top and readings are stored in picture coordinates, so they stay
fixed to the tree however far you zoom or pan.
"""
from __future__ import annotations

from .ruler import Point, Rect


class Viewport:
    MIN_ZOOM = 1.0      # 1.0 = whole picture fits the screen
    MAX_ZOOM = 12.0

    def __init__(self, image_w: float, image_h: float, view: Rect):
        if image_w <= 0 or image_h <= 0:
            raise ValueError("image size must be positive")
        self.iw, self.ih = float(image_w), float(image_h)
        self.view = view
        self.zoom = 1.0
        self.cx, self.cy = self.iw / 2, self.ih / 2   # picture point at the screen centre

    # ---- scale --------------------------------------------------------------
    @property
    def fit_scale(self) -> float:
        return min(self.view.w / self.iw, self.view.h / self.ih)

    @property
    def scale(self) -> float:
        """Screen pixels per picture pixel."""
        return self.fit_scale * self.zoom

    @property
    def zoom_percent(self) -> int:
        return round(self.zoom * 100)

    # ---- conversions ---------------------------------------------------------
    def to_screen(self, p: Point) -> Point:
        s = self.scale
        return Point(self.view.x + self.view.w / 2 + (p.x - self.cx) * s,
                     self.view.y + self.view.h / 2 - (p.y - self.cy) * s)

    def to_image(self, p: Point) -> Point:
        s = self.scale
        return Point(self.cx + (p.x - self.view.x - self.view.w / 2) / s,
                     self.cy - (p.y - self.view.y - self.view.h / 2) / s)

    def image_rect(self) -> Rect:
        """Where the whole picture is drawn on screen (may extend past the edges)."""
        bottom_left = self.to_screen(Point(0, self.ih))
        return Rect(bottom_left.x, bottom_left.y, self.iw * self.scale, self.ih * self.scale)

    def contains_image_point(self, p: Point) -> bool:
        return 0 <= p.x <= self.iw and 0 <= p.y <= self.ih

    # ---- gestures ------------------------------------------------------------
    def zoom_at(self, factor: float, anchor: Point) -> None:
        """Zoom by `factor`, keeping the picture point under `anchor` (screen) still."""
        fixed = self.to_image(anchor)
        self.zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self.zoom * factor))
        s = self.scale
        self.cx = fixed.x - (anchor.x - self.view.x - self.view.w / 2) / s
        self.cy = fixed.y + (anchor.y - self.view.y - self.view.h / 2) / s
        self._clamp()

    def zoom_centre(self, factor: float) -> None:
        self.zoom_at(factor, Point(self.view.x + self.view.w / 2, self.view.y + self.view.h / 2))

    def pan(self, dx: float, dy: float) -> None:
        """Move the picture by a screen distance (finger movement)."""
        s = self.scale
        self.cx -= dx / s
        self.cy += dy / s
        self._clamp()

    def reset(self) -> None:
        self.zoom = 1.0
        self.cx, self.cy = self.iw / 2, self.ih / 2

    def set_view(self, view: Rect) -> None:
        self.view = view
        self._clamp()

    def _clamp(self) -> None:
        """Stop the picture being dragged off screen; centre it if it's smaller than the view."""
        s = self.scale
        half_w, half_h = self.view.w / 2 / s, self.view.h / 2 / s
        self.cx = self.iw / 2 if self.iw <= 2 * half_w else min(max(self.cx, half_w), self.iw - half_w)
        self.cy = self.ih / 2 if self.ih <= 2 * half_h else min(max(self.cy, half_h), self.ih - half_h)
