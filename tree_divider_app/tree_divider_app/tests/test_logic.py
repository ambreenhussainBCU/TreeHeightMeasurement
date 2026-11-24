import math

import pytest
from PIL import Image

from tree_divider import exporter
from tree_divider.ruler import Point, Rect, Ruler, fit_contain, screen_to_image
from tree_divider.session import Session


# ---- Ruler ------------------------------------------------------------------
def test_reading_matches_screenshot_example():
    # Base at y=742, top at y=43 (image pixels, y down); tapped at y=660.
    r = Ruler(Point(360, 742), Point(343, 43), 100)
    assert r.reading(Point(383, 660)) == pytest.approx(11.7, abs=0.2)


def test_divisions_are_equal():
    r = Ruler(Point(0, 0), Point(0, 1000), 100)
    ticks = r.ticks()
    assert len(ticks) == 101
    gaps = {round((ticks[i + 1].start.y - ticks[i].start.y), 6) for i in range(100)}
    assert gaps == {10.0}


def test_tick_kinds():
    kinds = [t.kind for t in Ruler(Point(0, 0), Point(0, 100), 100).ticks()]
    assert kinds[0] == kinds[10] == kinds[100] == "major"
    assert kinds[5] == kinds[95] == "mid"
    assert kinds[1] == "minor"


def test_reading_projects_sideways_taps_onto_trunk_and_clamps():
    r = Ruler(Point(0, 0), Point(0, 200), 100)
    assert r.reading(Point(50, 100)) == pytest.approx(50)
    assert r.reading(Point(0, -40)) == 0
    assert r.reading(Point(0, 999)) == 100


def test_tilted_tree():
    r = Ruler(Point(0, 0), Point(100, 100), 10)
    assert r.reading(Point(50, 50)) == pytest.approx(5)
    assert r.distance_to_line(Point(0, 100)) == pytest.approx(math.sqrt(2) * 50)


def test_label_step_thins_numbers_to_fit():
    r = Ruler(Point(0, 0), Point(0, 700), 100)   # 7 px per part
    assert r.label_step(24) == 5
    labs = r.labels(24, 30)
    assert [l.value for l in labs][:3] == [5, 10, 15] and labs[-1].value == 100
    few = Ruler(Point(0, 0), Point(0, 700), 10)
    assert [l.value for l in few.labels(24, 30)] == list(range(1, 11))
    assert all(l.centred for l in few.labels(24, 30))


def test_bands_group_by_ten_when_many_parts():
    assert len(Ruler(Point(0, 0), Point(0, 100), 100).bands()) == 10
    assert len(Ruler(Point(0, 0), Point(0, 100), 10).bands()) == 10
    assert len(Ruler(Point(0, 0), Point(0, 100), 15).bands()) == 15


def test_invalid_ruler():
    with pytest.raises(ValueError):
        Ruler(Point(1, 1), Point(1, 1))
    with pytest.raises(ValueError):
        Ruler(Point(0, 0), Point(0, 1), 0)


# ---- Screen <-> image mapping ------------------------------------------------
def test_fit_contain_letterboxes():
    shown = fit_contain(Rect(0, 0, 400, 800), 1000, 1000)
    assert (shown.x, shown.y, shown.w, shown.h) == (0, 200, 400, 400)


def test_screen_to_image_flips_y():
    shown = Rect(0, 200, 400, 400)
    assert screen_to_image(Point(0, 200), shown, 1000, 1000) == Point(0, 1000)   # bottom-left
    assert screen_to_image(Point(400, 600), shown, 1000, 1000) == Point(1000, 0)  # top-right


# ---- Session ----------------------------------------------------------------
def test_tap_flow_base_top_then_reading():
    s = Session(parts=100, hit_radius=20, read_radius=50)
    assert s.stage == "base"
    assert s.press(Point(100, 0)) == "base_set"
    assert s.press(Point(100, 1000)) == "top_set"
    s.release()
    assert s.press(Point(110, 116.4)) == "reading_added"
    s.release()
    assert s.reading_value(0) == pytest.approx(11.64)


def test_far_taps_are_ignored():
    s = Session(read_radius=50)
    s.press(Point(0, 0)); s.press(Point(0, 1000)); s.release()
    assert s.press(Point(300, 500)) is None
    assert s.readings == []


def test_dragging_handles_and_readings():
    s = Session(parts=10, hit_radius=20)
    s.press(Point(0, 0)); s.press(Point(0, 100)); s.release()
    s.press(Point(0, 50)); s.release()                      # reading at 5.0
    assert s.press(Point(2, 99)) == "grabbed"               # grab top
    s.drag(Point(0, 200)); s.release()
    assert s.reading_value(0) == pytest.approx(5.0)         # reading moved with tree
    s.press(Point(0, 100)); s.drag(Point(0, 150)); s.release()
    assert s.reading_value(0) == pytest.approx(7.5)


def test_double_tap_removes_reading_and_parts_rescale():
    s = Session(parts=100)
    s.press(Point(0, 0)); s.press(Point(0, 1000)); s.release()
    s.press(Point(0, 300)); s.release()
    s.set_parts(10)
    assert s.reading_value(0) == pytest.approx(3.0)
    assert s.press(Point(0, 300), double_tap=True) == "reading_removed"
    assert s.readings == []


def test_parts_stepping_limits():
    s = Session(parts=100)
    s.step_parts(1); assert s.parts == 110
    s.set_parts(20); s.step_parts(-1); assert s.parts == 19
    s.set_parts(20); s.step_parts(1); assert s.parts == 30
    s.set_parts(1); assert s.parts == 2
    s.set_parts(999); assert s.parts == 200


# ---- Pillow exporter ---------------------------------------------------------
def test_export_draws_on_picture(tmp_path):
    img = Image.new("RGB", (600, 800), (40, 90, 40))
    path = exporter.save(img, Point(300, 760), Point(300, 40), 100, [0.1164], str(tmp_path))
    assert path.endswith(".jpg") and "reading-11_64" in path
    out = Image.open(path)
    assert out.size == (600, 800)
    assert out.getpixel((300, 400)) != (40, 90, 40)   # ruler drawn
    assert out.getpixel((20, 20))[1] > 80             # background untouched


def test_image_from_rgba_flips():
    top_red_bottom_blue = bytes([0, 0, 255, 255] * 2 + [255, 0, 0, 255] * 2)  # GL: bottom row first
    img = exporter.image_from_rgba(top_red_bottom_blue, 2, 2)
    assert img.getpixel((0, 0))[:3] == (255, 0, 0)


def test_zoomed_in_100_parts_labels_mark_positions_not_parts():
    r = Ruler(Point(0, 0), Point(0, 5000), 100)     # 50 px per part: room for every number
    labs = r.labels(24, 30)
    assert r.label_step(24) == 1
    assert [l.value for l in labs][:3] == [1, 2, 3]
    assert not any(l.centred for l in labs)
    assert labs[10].anchor.y == pytest.approx(r.point_at(0.11).y)   # "11" sits at 11 parts
