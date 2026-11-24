"""Machine-vision extraction, tested on pictures made by the app's own exporter."""
import os
import shutil
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
cv2 = pytest.importorskip("cv2")
from extract_reading import extract, parse_reading  # noqa: E402

from tree_divider import exporter  # noqa: E402
from tree_divider.ruler import Point  # noqa: E402

HAS_OCR = shutil.which("tesseract") is not None


def make_picture(tmp_path, size, base, top, parts, readings, name="p.jpg"):
    bg = Image.new("RGB", size, (150, 175, 160))
    path = str(tmp_path / name)
    exporter.render(bg, base, top, parts, readings).save(path, "JPEG", quality=85)
    return path


@pytest.mark.parametrize("size,base,top,parts,reads", [
    ((611, 791), Point(357, 741), Point(344, 48), 100, [0.1153]),
    ((1200, 900), Point(500, 850), Point(650, 60), 100, [0.25, 0.8]),     # tilted, two readings
    ((900, 1600), Point(450, 1500), Point(450, 120), 10, [0.437]),        # big picture, 10 parts
])
def test_geometric_reading(tmp_path, size, base, top, parts, reads):
    res = extract(make_picture(tmp_path, size, base, top, parts, reads), parts)
    got = sorted(r.geometric for r in res.readings)
    want = sorted(f * parts for f in reads)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= parts * 0.004          # within 0.4% of the tree height


@pytest.mark.skipif(not HAS_OCR, reason="Tesseract not installed")
def test_banner_text_is_read_and_confirmed(tmp_path):
    res = extract(make_picture(tmp_path, (611, 791), Point(357, 741), Point(344, 48), 100, [0.1153]))
    r = res.readings[0]
    assert r.ocr == pytest.approx(11.53) and r.agree
    assert r.value == pytest.approx(11.53) and r.source.startswith("banner")


def test_parse_uses_two_decimals():
    assert parse_reading("11.53") == 11.53
    assert parse_reading("1153") == 11.53       # decimal point lost by OCR
    assert parse_reading("0.4.3") == 0.43
    assert parse_reading("7") is None


def test_no_ruler_raises(tmp_path):
    path = str(tmp_path / "blank.jpg")
    Image.new("RGB", (300, 400), (90, 120, 90)).save(path)
    with pytest.raises(ValueError):
        extract(path)
