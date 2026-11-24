"""App-side machine vision: check readings after a dot is added; find rulers in opened pictures."""
import os
import shutil

import pytest
from PIL import Image

pytest.importorskip("cv2")
from tree_divider import exporter, vision_check  # noqa: E402
from tree_divider.ruler import Point  # noqa: E402

HAS_OCR = shutil.which("tesseract") is not None
REAL = os.path.join(os.path.dirname(__file__), "data", "real-reading-11_53.jpg")   # a real saved picture


def photo(size=(900, 1600)):
    img = Image.new("RGB", size, (140, 170, 150))
    for x in range(0, size[0], 37):                       # some texture
        img.paste((120, 150, 130), (x, 0, x + 9, size[1]))
    return img


@pytest.mark.skipif(not HAS_OCR, reason="Tesseract not installed")
def test_check_confirms_each_reading():
    pic = photo()
    checks = vision_check.check_readings(pic, Point(450, 1500), Point(430, 100), 100, [0.1153, 0.62])
    assert [c.status for c in checks] == ["ok", "ok"]
    assert checks[0].text == "Vision 11.53"


def test_check_without_ocr_still_measures():
    checks = vision_check.check_readings(photo(), Point(450, 1500), Point(430, 100), 100, [0.3],
                                         use_ocr=False)
    assert checks[0].status == "measured"
    assert abs(float(checks[0].text.split()[1]) - 30.0) <= 0.4


def test_check_big_photo_is_shrunk_but_still_correct():
    big = photo((3000, 4000))
    checks = vision_check.check_readings(big, Point(1500, 3800), Point(1500, 200), 100, [0.5],
                                         use_ocr=False)
    assert abs(float(checks[0].text.split()[1]) - 50.0) <= 0.4


@pytest.mark.skipif(not HAS_OCR, reason="Tesseract not installed")
def test_finds_ruler_in_saved_picture_and_guesses_parts():
    pic = exporter.render(photo(), Point(450, 1500), Point(450, 100), 50, [0.4])
    found = vision_check.find_existing_ruler(pic, parts=100)    # app is set to 100, picture used 50
    assert found is not None and found.parts == 50 and found.parts_guessed
    assert found.values == [20.0]
    assert abs(found.base.y - 1500) < 6 and abs(found.top.y - 100) < 6


@pytest.mark.skipif(not HAS_OCR, reason="Tesseract not installed")
def test_finds_ruler_in_users_real_picture():
    if not os.path.exists(REAL):
        pytest.skip("real picture not available")
    found = vision_check.find_existing_ruler(Image.open(REAL), parts=100)
    assert found and found.values == [11.53] and found.parts == 100


def test_plain_photo_has_no_ruler():
    assert vision_check.find_existing_ruler(photo(), 100) is None


@pytest.mark.skipif(not HAS_OCR, reason="Tesseract not installed")
def test_loaded_reading_equals_banner_text_exactly():
    if not os.path.exists(REAL):
        pytest.skip("real picture not available")
    found = vision_check.find_existing_ruler(Image.open(REAL), parts=100)
    assert round(found.fractions[0] * found.parts, 2) == 11.53
