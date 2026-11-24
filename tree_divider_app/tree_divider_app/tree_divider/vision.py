"""
Machine vision: read the blue banner reading (e.g. 11.53) from a Tree Divider picture.

Used by the app (automatic check after you add a reading, and detecting rulers in
opened pictures) and by tools/extract_reading.py (command line, folders, CSV).

Two independent methods, which check each other:

  1. GEOMETRY  - find the ruler, the Base and Top dots and the blue reading
                 dot by colour, then measure how far up the ruler the blue dot is:
                     reading = (dot - base) . (top - base) / |top - base|^2 * parts
                 This is the same formula the app uses, so it recovers the value
                 even if the banner text is blurred or cut off.

  2. OCR       - crop the blue banner and read its digits with Tesseract.

Final value: the banner text when the measurement confirms it (exact), otherwise
the measurement, flagged "CHECK" (OCR misread a digit or the banner was hidden).

Needs: opencv-python, numpy, and (for OCR) pytesseract + the Tesseract program.
"""
from __future__ import annotations

import os
import shutil
import re
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# HSV ranges (OpenCV: H 0-179, S/V 0-255). The ruler bands are slightly see-through,
# so the ranges are generous; the blue is far more saturated than sky.
YELLOW = ((15, 90, 140), (35, 255, 255))
PINK = ((155, 80, 110), (179, 255, 255))
BLUE = ((95, 140, 160), (115, 255, 255))


@dataclass
class Reading:
    value: float                  # final answer
    source: str                   # "banner text (confirmed)" or "geometry"
    geometric: float              # from dot positions
    ocr: float | None = None      # from the banner text
    ocr_text: str = ""
    dot_xy: tuple = ()            # pixel position of the blue dot
    banner_box: tuple = ()        # x, y, w, h of the banner
    agree: bool | None = None


@dataclass
class Result:
    parts: int
    base_xy: tuple
    top_xy: tuple
    ruler_length_px: float
    readings: list = field(default_factory=list)
    fractions: list = field(default_factory=list)   # reading positions 0..1, same order


# ---------------------------------------------------------------- colour masks
def colour_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rng = lambda r: cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))  # noqa: E731
    return rng(YELLOW), rng(PINK), rng(BLUE)


# ---------------------------------------------------------------- the ruler
def find_ruler(yellow, pink):
    """Return (point_on_line, unit_direction, half_width, ruler_mask)."""
    mask = cv2.bitwise_or(yellow, pink)
    # Close the gaps the white tick marks cut across the coloured band.
    # Ticks get thicker on bigger pictures, so the closing size scales with the picture.
    k = max(5, int(round(max(mask.shape) * 0.006)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise ValueError("no yellow/pink ruler found")

    # The longest thin component is the ruler (labels and skin are short blobs).
    best, best_len = None, 0.0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        length, width = max(w, h), max(1.0, min(w, h))
        if length / width > 4 and length > best_len:
            best, best_len = i, length
    if best is None:
        raise ValueError("no long ruler-shaped stripe found")

    pts = np.column_stack(np.where(labels == best))[:, ::-1].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    origin, direction = np.array([x0, y0]), np.array([vx, vy])
    normal = np.array([-vy, vx])

    # Add the other pieces of the ruler (the blue dot splits it in two).
    all_pts = np.column_stack(np.where(mask > 0))[:, ::-1].astype(np.float32)
    perp = np.abs((all_pts - origin) @ normal)
    core_perp = np.abs((pts - origin) @ normal)
    half_width = float(np.percentile(core_perp, 90))
    on_line = all_pts[perp <= half_width * 1.8]
    ruler_mask = np.zeros_like(mask)
    ruler_mask[on_line[:, 1].astype(int), on_line[:, 0].astype(int)] = 255
    return origin, direction, half_width, ruler_mask


def find_end_dots(ruler_mask, origin, direction, yellow, pink):
    """Centres of the Base and Top dots: the widest spots at each end of the ruler."""
    dist = cv2.distanceTransform(ruler_mask, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(ruler_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    t = (pts - origin) @ direction
    t_min, t_max = t.min(), t.max()
    window = (t_max - t_min) * 0.06

    def centre(sel):
        cand = pts[sel]
        d = dist[cand[:, 1].astype(int), cand[:, 0].astype(int)]
        top = cand[d >= d.max() * 0.9]          # the fattest part = the dot
        return top.mean(axis=0)

    a, b = centre(t <= t_min + window), centre(t >= t_max - window)

    def colour_at(p, r=4):
        x, y = int(round(p[0])), int(round(p[1]))
        ys_, xs_ = slice(max(0, y - r), y + r + 1), slice(max(0, x - r), x + r + 1)
        return yellow[ys_, xs_].mean(), pink[ys_, xs_].mean()

    ya, pa = colour_at(a)
    yb, pb = colour_at(b)
    if ya - pa > yb - pb:          # the yellow end is the Base
        return a, b
    if yb - pb > ya - pa:
        return b, a
    return (a, b) if a[1] > b[1] else (b, a)   # tie: base is lower in the picture


# ---------------------------------------------------------------- blue reading dots
def find_reading_dots(blue, base, top, half_width):
    """Blue pixels sitting on the ruler line = reading dots (connector/banner are off to the side)."""
    axis = top - base
    length = np.linalg.norm(axis)
    unit = axis / length
    normal = np.array([-unit[1], unit[0]])
    ys, xs = np.nonzero(blue)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    perp = np.abs((pts - base) @ normal)
    near = pts[perp <= half_width * 2.2]
    if len(near) == 0:
        return []
    m = np.zeros_like(blue)
    m[near[:, 1].astype(int), near[:, 0].astype(int)] = 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    dots = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            continue
        comp = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        # Use only the part closest to the line so the sideways connector doesn't pull it.
        c_perp = np.abs((comp - base) @ normal)
        core = comp[c_perp <= half_width * 1.2]
        dots.append(((core if len(core) else comp).mean(axis=0), len(comp)))
    # Tick marks can cut one dot into pieces: merge pieces at (almost) the same height.
    dots.sort(key=lambda d: float((d[0] - base) @ unit))
    merged = []
    for p, w in dots:
        if merged and abs(float((p - merged[-1][0]) @ unit)) < half_width * 2.5:
            q, wq = merged[-1]
            merged[-1] = ((q * wq + p * w) / (wq + w), wq + w)
        else:
            merged.append((p, w))
    return [p for p, _ in merged]


# ---------------------------------------------------------------- OCR of the banner
def find_banners(blue, min_area=150):
    # Cut the thin connector line that joins the banner to the dot, so the banner
    # box holds only the banner (opening wider than the line, narrower than the banner).
    k = max(3, int(round(max(blue.shape) * 0.006)) | 1)
    opened = cv2.morphologyEx(blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and w > 1.6 * h:   # wide rounded pill
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _prepare_banner(bgr, box, binary=False):
    """Dark digits on white, banner scaled to ~100 px tall, ready for Tesseract.

    (Settings chosen by measurement on 78 test banners: single-line mode on this
    image reads 76/78 correctly.)
    """
    x, y, w, h = box
    pad = max(1, h // 8)
    crop = bgr[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    red = crop[:, :, 2]                       # white text: red=255, blue banner: red~40
    scale = 100.0 / max(1, h)
    red = cv2.resize(red, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, text = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Bright areas outside the banner (sky etc.) touch the crop edge: drop them.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text)
    H, W = text.shape
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        if bx == 0 or by == 0 or bx + bw >= W or by + bh >= H:
            text[labels == i] = 0
    if binary:
        img = 255 - text
    else:                                     # keep the soft grey edges of the digits
        img = 255 - red
        img[text == 0] = 255
    return cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def parse_reading(raw):
    """The app always prints two decimals, so the digits alone fix the value: '1153' -> 11.53."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 3:
        return None
    return int(digits) / 100


_TESSERACT_GUESSES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
)


def _tesseract():
    """pytesseract, pointed at the Tesseract program, or None if it isn't installed."""
    try:
        import pytesseract
    except ImportError:
        return None
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        cmd = next((p for p in _TESSERACT_GUESSES if os.path.exists(p)), None)
    if not cmd:
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def ocr_available() -> bool:
    return _tesseract() is not None


def ocr_banner(bgr, box):
    pytesseract = _tesseract()
    if pytesseract is None:
        return None, ""
    raw = ""
    for binary in (False, True):              # grey first; plain black/white as a backup
        raw = pytesseract.image_to_string(
            _prepare_banner(bgr, box, binary),
            config="--psm 7 -c tessedit_char_whitelist=0123456789.").strip()
        value = parse_reading(raw)
        if value is not None:
            return value, raw
    return None, raw


# ---------------------------------------------------------------- main routine
def extract(path, parts=100, debug_path=None, tolerance=None, use_ocr=True):
    """Find every reading in a picture file."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return analyse(bgr, parts, debug_path, tolerance, use_ocr)


def analyse(bgr, parts=100, debug_path=None, tolerance=None, use_ocr=True):
    """Find every reading in a BGR picture (numpy array, as OpenCV uses).

    `tolerance` (in parts) for OCR vs geometry defaults to 1% of the tree height,
    about 7x the typical geometric error.
    """
    if tolerance is None:
        tolerance = max(0.05, parts * 0.01)
    yellow, pink, blue = colour_masks(bgr)
    origin, direction, half_width, ruler_mask = find_ruler(yellow, pink)
    base, top = find_end_dots(ruler_mask, origin, direction, yellow, pink)
    axis = top - base
    length = float(np.linalg.norm(axis))

    # Exclude the blue dot area when looking for banners (the dot is round, not wide)
    banners = find_banners(blue)
    result = Result(parts, tuple(np.round(base, 1).tolist()), tuple(np.round(top, 1).tolist()),
                    round(length, 1))

    for dot in find_reading_dots(blue, base, top, half_width):
        frac = float((dot - base) @ axis / length ** 2)
        result.fractions.append(frac)
        geo = round(frac * parts, 2)
        r = Reading(geo, "geometry", geo, dot_xy=tuple(np.round(dot, 1).tolist()))
        # The banner belongs to this dot if its centre projects to the same height on the ruler.
        best = None
        for box in banners:
            c = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
            gap = abs(float((c - dot) @ axis / length))
            if gap < box[3] * 1.5 and (best is None or gap < best[0]):
                best = (gap, box)
        if best:
            r.banner_box = best[1]
            if use_ocr:
                r.ocr, r.ocr_text = ocr_banner(bgr, best[1])
            if r.ocr is not None:
                r.agree = abs(r.ocr - r.geometric) <= tolerance
                if r.agree:      # exact printed value, confirmed by the measurement
                    r.value, r.source = r.ocr, "banner text (confirmed)"
        result.readings.append(r)

    if debug_path:
        dbg = bgr.copy()
        cv2.line(dbg, tuple(map(int, base)), tuple(map(int, top)), (0, 255, 0), 1)
        for p, c in ((base, (0, 200, 255)), (top, (180, 60, 255))):
            cv2.drawMarker(dbg, tuple(map(int, p)), c, cv2.MARKER_CROSS, 22, 2)
        for r in result.readings:
            cv2.drawMarker(dbg, tuple(map(int, r.dot_xy)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            if r.banner_box:
                x, y, w, h = r.banner_box
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"geo {r.geometric:.2f} | ocr {r.ocr}"
            pos = (10, 30 + 28 * result.readings.index(r))
            cv2.putText(dbg, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
            cv2.putText(dbg, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
        cv2.imwrite(debug_path, dbg)
    return result


def pil_to_bgr(img):
    """Pillow image -> OpenCV BGR array."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
