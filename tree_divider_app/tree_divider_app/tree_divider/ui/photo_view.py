"""
PhotoView: shows the picture (camera frame or photo) with the ruler on top,
and turns touches into actions.

Gestures
  tap                    set base / top / add a reading
  drag a dot             move base, top or a reading
  double-tap a reading   remove it
  drag empty space       pan (when zoomed in)
  two-finger pinch       zoom in/out and pan
  mouse wheel            zoom around the pointer
  right/middle drag      pan (computer)
"""
from __future__ import annotations

from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.stencilview import StencilView

from ..ruler import Point, Rect, Ruler
from ..viewport import Viewport

YELLOW = (1, 0.824, 0.247)
PINK = (1, 0.239, 0.545)
BLUE = (0.161, 0.627, 1)
INK = (0.106, 0.09, 0.071)
PAPER = (0.969, 0.961, 0.937)
WHITE = (1, 1, 1)
GREEN = (0.56, 0.92, 0.56)
ORANGE = (1, 0.55, 0.2)

# Vision label colours: status -> (background, text, background alpha)
VISION_STYLE = {
    "ok": (INK, GREEN, 0.9),
    "measured": (INK, PAPER, 0.9),
    "check": (ORANGE, WHITE, 1.0),
    "not_found": (INK, ORANGE, 0.9),
    "pending": (INK, PAPER, 0.6),
}

WHEEL_STEP = 1.2
TAP_SLOP = dp(10)       # finger movement allowed before a tap becomes a pan


class PhotoView(StencilView):
    session = ObjectProperty(None)
    on_event = ObjectProperty(None)        # callback(event_name)
    zoom_percent = NumericProperty(100)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viewport: Viewport | None = None
        self.texture = None
        self.vision_labels = []    # one vision_check.Check per reading (or empty)
        self.height_labels = []    # one note per reading: "Banner 1 m", "3.92 m above base"
        self._text_cache = {}
        self._touches = {}         # uid -> touch, all fingers currently down on us
        self._mode = {}            # uid -> "tap" | "drag" | "pan" | "pinch" | "done"
        self._start = {}           # uid -> start position
        self._pinch_prev = None    # previous positions of the two pinch fingers
        self.bind(size=self._on_resize, pos=self._on_resize)

    # ---- picture ---------------------------------------------------------------
    def set_picture(self, texture, width, height):
        """Show a new picture. Resets zoom if its size changed."""
        self.texture = texture
        vp = self.viewport
        if vp is None or (vp.iw, vp.ih) != (float(width), float(height)):
            self.viewport = Viewport(width, height, self._rect())
        self._sync()

    def _rect(self):
        return Rect(self.x, self.y, max(1, self.width), max(1, self.height))

    def _on_resize(self, *_):
        if self.viewport:
            self.viewport.set_view(self._rect())
        self._sync()

    # ---- zoom controls (buttons) --------------------------------------------------
    def zoom_by(self, factor):
        if self.viewport:
            self.viewport.zoom_centre(factor)
            self._sync()

    def fit(self):
        if self.viewport:
            self.viewport.reset()
            self._sync()

    # ---- helpers --------------------------------------------------------------------
    def _img(self, pos) -> Point:
        return self.viewport.to_image(Point(*pos))

    def _scr(self, p: Point) -> Point:
        return self.viewport.to_screen(p)

    def _notify(self, event):
        if event and self.on_event:
            self.on_event(event)

    def _sync(self):
        if self.viewport and self.session:
            self.session.scale = self.viewport.scale
            self.zoom_percent = self.viewport.zoom_percent
        self.redraw()

    # ---- touches ----------------------------------------------------------------------
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) or self.viewport is None:
            return False
        if touch.is_mouse_scrolling:
            if touch.button in ("scrolldown", "scrollup"):
                factor = WHEEL_STEP if touch.button == "scrolldown" else 1 / WHEEL_STEP
                self.viewport.zoom_at(factor, Point(*touch.pos))
                self._sync()
            return True

        touch.grab(self)
        uid = touch.uid
        self._touches[uid] = touch
        self._start[uid] = touch.pos

        if len(self._touches) >= 2:
            # Second finger: everything becomes a pinch; cancel taps and drags.
            self.session.release()
            for other in self._touches:
                self._mode[other] = "pinch"
            ids = list(self._touches)[:2]
            self._pinch_prev = tuple(Point(*self._touches[i].pos) for i in ids)
            return True

        if getattr(touch, "button", "left") in ("right", "middle"):
            self._mode[uid] = "pan"
            return True

        p = self._img(touch.pos)
        target = self.session.target_at(p)
        if target is not None:
            if touch.is_double_tap and isinstance(target, tuple):
                self.session.delete_reading(target[1])
                self._mode[uid] = "done"
                self._notify("reading_removed")
            else:
                self.session.grab(p)
                self._mode[uid] = "drag"
        else:
            self._mode[uid] = "tap"
        self.redraw()
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return False
        uid = touch.uid
        mode = self._mode.get(uid)

        if mode == "pinch" and len(self._touches) >= 2 and self._pinch_prev:
            ids = list(self._touches)[:2]
            a, b = (Point(*self._touches[i].pos) for i in ids)
            pa, pb = self._pinch_prev
            old_mid = Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
            new_mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
            old_d, new_d = pa.dist(pb), a.dist(b)
            self.viewport.pan(new_mid.x - old_mid.x, new_mid.y - old_mid.y)
            if old_d > dp(4):
                self.viewport.zoom_at(new_d / old_d, new_mid)
            self._pinch_prev = (a, b)
            self._sync()
            return True

        if mode == "tap":
            sx, sy = self._start[uid]
            if Point(sx, sy).dist(Point(*touch.pos)) > TAP_SLOP:
                self._mode[uid] = mode = "pan"
        if mode == "pan":
            self.viewport.pan(touch.dx, touch.dy)
            self._sync()
        elif mode == "drag":
            self.session.drag(self._img(touch.pos))
            self.redraw()
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        touch.ungrab(self)
        uid = touch.uid
        mode = self._mode.pop(uid, None)
        self._touches.pop(uid, None)
        self._start.pop(uid, None)

        if mode == "tap":
            p = self._img(touch.pos)
            if self.viewport.contains_image_point(p):
                self._notify(self.session.tap(p))
        elif mode == "drag":
            self.session.release()
            self._notify("moved")
        if len(self._touches) < 2:
            self._pinch_prev = None
            for other in self._touches:     # the finger left after a pinch shouldn't tap
                self._mode[other] = "done"
        self.redraw()
        return True

    # ---- drawing ------------------------------------------------------------------------
    def _text(self, text, size, colour):
        key = (text, size, colour)
        if key not in self._text_cache:
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            lab = CoreLabel(text=text, font_size=size, bold=True, color=colour + (1,))
            lab.refresh()
            self._text_cache[key] = lab.texture
        return self._text_cache[key]

    def _pill(self, text, centre, bg, fg, alpha=1.0, size=sp(14), extra=0):
        """Rounded label; `extra` adds room on the right (for a drawn icon). Returns its box."""
        tex = self._text(text, size, fg)
        pad_x, pad_y = dp(7), dp(3)
        w, h = tex.width + pad_x * 2 + extra, tex.height + pad_y * 2
        x, y = centre.x - w / 2, centre.y - h / 2
        Color(*bg, alpha)
        RoundedRectangle(pos=(x, y), size=(w, h), radius=[h / 2])
        Color(1, 1, 1, 1)
        Rectangle(texture=tex, pos=(x + pad_x, y + pad_y), size=tex.size)
        return x, y, w, h

    def _dot(self, p, radius, colour):
        ring = radius + dp(2.5)
        Color(*WHITE, 1)
        Ellipse(pos=(p.x - ring, p.y - ring), size=(ring * 2, ring * 2))
        Color(*colour, 1)
        Ellipse(pos=(p.x - radius, p.y - radius), size=(radius * 2, radius * 2))

    def redraw(self):
        self.canvas.clear()
        vp, s = self.viewport, self.session
        with self.canvas:
            Color(0, 0, 0, 1)
            Rectangle(pos=self.pos, size=self.size)
            if vp is None or s is None:
                return
            if self.texture is not None:
                r = vp.image_rect()
                Color(1, 1, 1, 1)
                Rectangle(texture=self.texture, pos=(r.x, r.y), size=(r.w, r.h))

            ruler = None
            if s.base is not None and s.top is not None:
                a, b = self._scr(s.base), self._scr(s.top)
                if a.dist(b) > 0.5:
                    ruler = Ruler(a, b, s.parts)       # same ruler, in screen pixels
            if ruler:
                for band in ruler.bands():
                    Color(*(PINK if band.alternate else YELLOW), 0.85)
                    Line(points=[band.start.x, band.start.y, band.end.x, band.end.y],
                         width=dp(7), cap="none")
                fine = dp(1) if ruler.part_length > dp(12) else dp(0.6)   # bolder when zoomed in
                for tick in ruler.ticks(dp(38), dp(28), dp(21)):
                    Color(*WHITE, 1)
                    Line(points=[tick.start.x, tick.start.y, tick.end.x, tick.end.y],
                         width=dp(1.3) if tick.kind == "major" else fine)
                # Only draw labels that are on screen (lots of them when zoomed in).
                box_pad = dp(40)
                for lab in ruler.labels(min_spacing=dp(22), offset=dp(38)):
                    q = lab.anchor
                    if not (self.x - box_pad <= q.x <= self.right + box_pad and
                            self.y - box_pad <= q.y <= self.top + box_pad):
                        continue
                    if lab.centred:
                        bg, fg = (PINK, WHITE) if lab.value % 2 == 0 else (YELLOW, INK)
                        self._pill(str(lab.value), q, bg, fg)
                    else:
                        self._pill(str(lab.value), q, INK,
                                   YELLOW if lab.major else PAPER, 1.0 if lab.major else 0.75)
                dragging = s.dragging is not None
                for i, f in enumerate(s.readings):
                    p = ruler.point_at(f)
                    bubble = p + ruler.normal.scale(-ruler.side * dp(60))
                    Color(*BLUE, 1)
                    Line(points=[p.x, p.y, bubble.x, bubble.y], width=dp(1))
                    self._dot(p, dp(8), BLUE)
                    self._pill(f"{s.reading_value(i):.2f}", bubble, BLUE, WHITE, size=sp(16))
                    # Notes stack below the banner; near the base (where the Base tag sits)
                    # they stack upwards instead so they don't cover it.
                    base_scr = ruler.point_at(0)
                    step = dp(24) if abs(bubble.y - base_scr.y) > dp(80) else -dp(24)
                    below = bubble.y - (dp(26) if step > 0 else -dp(26))
                    # Height in metres, next to the banner.
                    if i < len(self.height_labels) and self.height_labels[i]:
                        self._pill(self.height_labels[i], Point(bubble.x, below),
                                   YELLOW if i == 0 else INK, INK if i == 0 else PAPER,
                                   0.95, size=sp(12))
                        below -= step
                    # Machine vision's own reading of this banner, below that.
                    if not dragging and i < len(self.vision_labels) and self.vision_labels[i]:
                        check = self.vision_labels[i]
                        bg, fg, alpha = VISION_STYLE.get(check.status, VISION_STYLE["measured"])
                        tick = check.status == "ok"
                        text = check.text + ("  !" if check.status == "check" else "")
                        x, y, w, h = self._pill(text, Point(bubble.x, below), bg, fg,
                                                alpha, size=sp(12), extra=dp(14) if tick else 0)
                        if tick:      # drawn tick mark (the default font has no check-mark glyph)
                            cx, cy = x + w - dp(13), y + h / 2
                            Color(*GREEN, 1)
                            Line(points=[cx - dp(4), cy, cx - dp(1), cy - dp(3.5), cx + dp(5), cy + dp(4)],
                                 width=dp(1.2), cap="round", joint="round")

            away = ruler.normal.scale(-ruler.side * dp(44)) if ruler else Point(-dp(44), 0)
            for p_img, colour, name in ((s.base, YELLOW, "Base"), (s.top, PINK, "Top")):
                if p_img is not None:
                    p = self._scr(p_img)
                    self._dot(p, dp(12), colour)
                    self._pill(name, p + away, INK, PAPER)
