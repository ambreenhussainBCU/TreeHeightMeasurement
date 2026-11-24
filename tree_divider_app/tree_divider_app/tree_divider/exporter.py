"""
Save the picture with the ruler and readings drawn on it, using Pillow.

Pillow functions used:
  Image.open / Image.frombytes / Image.transpose / Image.convert / Image.save
  ImageDraw.Draw, draw.line, draw.ellipse, draw.rounded_rectangle, draw.text,
  draw.textbbox
  ImageFont.load_default(size=...)  (scalable default font, Pillow >= 10.1)
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .ruler import Point, Ruler

YELLOW = (255, 210, 63)
PINK = (255, 61, 139)
BLUE = (41, 160, 255)
INK = (27, 23, 18)
PAPER = (247, 245, 239)
WHITE = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


def _pill(draw: ImageDraw.ImageDraw, text: str, centre: Point, font, bg, fg, pad: int) -> None:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = (r - l) + pad * 2, (b - t) + pad * 2
    box = (centre.x - w / 2, centre.y - h / 2, centre.x + w / 2, centre.y + h / 2)
    draw.rounded_rectangle(box, radius=h / 2, fill=bg)
    draw.text((centre.x, centre.y), text, font=font, fill=fg, anchor="mm")


def render(image: Image.Image, base: Point, top: Point, parts: int,
           readings: Iterable[float] = (), labels: Sequence[str] = (),
           caption: str = "") -> Image.Image:
    """Return a copy of `image` with the ruler drawn on it.

    `base` and `top` are in image pixels (origin top-left).
    `readings` are fractions 0..1 along the ruler.
    `labels` (optional) are small notes under each reading's banner, e.g. "3.92 m".
    `caption` (optional) goes in a dark strip added below the picture, e.g. "Tree height 8.67 m".
    """
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    s = max(out.width, out.height) / 800          # scale sizes to the picture
    ruler = Ruler(base, top, parts)
    font = _font(max(10, round(15 * s)))

    for band in ruler.bands():
        colour = PINK if band.alternate else YELLOW
        draw.line([(band.start.x, band.start.y), (band.end.x, band.end.y)],
                  fill=colour + (220,), width=max(4, round(16 * s)))

    for tick in ruler.ticks(38 * s, 28 * s, 21 * s):
        width = max(1, round((2.5 if tick.kind == "major" else 1.2) * s))
        draw.line([(tick.start.x, tick.start.y), (tick.end.x, tick.end.y)], fill=WHITE, width=width)

    for lab in ruler.labels(min_spacing=24 * s, offset=38 * s):
        if lab.centred:
            bg, fg = (PINK, WHITE) if lab.value % 2 == 0 else (YELLOW, INK)
        else:
            bg, fg = (INK, YELLOW) if lab.major else (INK + (180,), PAPER)
        _pill(draw, str(lab.value), lab.anchor, font, bg, fg, round(5 * s))

    labels = list(labels)
    small = _font(max(9, round(12 * s)))
    for i, f in enumerate(readings):
        p = ruler.point_at(f)
        r = 9 * s
        draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), fill=BLUE, outline=WHITE, width=max(2, round(3 * s)))
        bubble = p + ruler.normal.scale(-ruler.side * 60 * s)
        draw.line([(p.x, p.y), (bubble.x, bubble.y)], fill=BLUE, width=max(1, round(2 * s)))
        _pill(draw, f"{f * parts:.2f}", bubble, font, BLUE, WHITE, round(6 * s))
        if i < len(labels) and labels[i]:
            # below the banner, or above it when close to the base (image y grows downwards)
            near_base = abs(bubble.y - ruler.point_at(0).y) <= 80 * s
            _pill(draw, labels[i], bubble + Point(0, -26 * s if near_base else 26 * s),
                  small, INK + (230,), PAPER, round(5 * s))

    for p, colour, name in ((base, YELLOW, "Base"), (top, PINK, "Top")):
        r = 13 * s
        draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), fill=colour, outline=WHITE, width=max(2, round(3 * s)))
        _pill(draw, name, p + ruler.normal.scale(-ruler.side * 48 * s), font, INK, PAPER, round(5 * s))
    if caption:
        # A dark strip added below the picture, so the caption never covers the ruler.
        big = _font(max(12, round(20 * s)))
        l, t, r, b = draw.textbbox((0, 0), caption, font=big)
        pad = round(12 * s)
        strip = (b - t) + 2 * pad
        framed = Image.new("RGB", (out.width, out.height + strip), INK)
        framed.paste(out, (0, 0))
        ImageDraw.Draw(framed).text((pad, out.height + strip / 2), caption, font=big,
                                    fill=YELLOW, anchor="lm")
        out = framed
    return out


def image_from_rgba(pixels: bytes, width: int, height: int, flip: bool = True) -> Image.Image:
    """Build a Pillow image from raw RGBA bytes (e.g. a Kivy camera texture).

    OpenGL textures are stored bottom row first, so they need flipping.
    """
    img = Image.frombytes("RGBA", (width, height), pixels)
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if flip else img


def file_name(parts: int, readings: Iterable[float], tree_m: Optional[float] = None) -> str:
    readings = list(readings)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pick = 0 if tree_m is not None else -1          # with a height: the banner (first) reading
    tag = f"-reading-{readings[pick] * parts:.2f}".replace(".", "_") if readings else ""
    if tree_m is not None:
        tag = f"-height-{tree_m:.2f}m".replace(".", "_") + tag
    return f"tree-divider{tag}-{stamp}.jpg"


def to_jpeg(image: Image.Image, base: Point, top: Point, parts: int,
            readings: Iterable[float], labels: Sequence[str] = (), caption: str = "") -> bytes:
    """The marked-up picture as JPEG bytes (Pillow `Image.save` into memory)."""
    buf = io.BytesIO()
    render(image, base, top, parts, list(readings), labels, caption).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def save(image: Image.Image, base: Point, top: Point, parts: int,
         readings: Iterable[float], folder: str, name: Optional[str] = None) -> str:
    """Save straight into `folder` (used by tests and scripts)."""
    readings = list(readings)
    path = os.path.join(folder, name or file_name(parts, readings))
    with open(path, "wb") as f:
        f.write(to_jpeg(image, base, top, parts, readings))
    return path
