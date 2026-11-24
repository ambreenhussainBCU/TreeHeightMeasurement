import pytest

from tree_divider.ruler import Point, Rect
from tree_divider.session import Session
from tree_divider.viewport import Viewport


def approx_pt(a, b, tol=1e-6):
    return a.x == pytest.approx(b.x, abs=tol) and a.y == pytest.approx(b.y, abs=tol)


def test_fit_centres_picture_and_flips_y():
    vp = Viewport(1000, 2000, Rect(0, 0, 400, 400))       # tall photo in square view
    assert vp.fit_scale == 0.2
    r = vp.image_rect()
    assert (r.x, r.y, r.w, r.h) == (100, 0, 200, 400)
    assert approx_pt(vp.to_screen(Point(0, 0)), Point(100, 400))       # top-left of photo
    assert approx_pt(vp.to_screen(Point(1000, 2000)), Point(300, 0))   # bottom-right


def test_round_trip_at_any_zoom():
    vp = Viewport(4000, 3000, Rect(10, 20, 390, 700))
    vp.zoom_at(3.7, Point(150, 300))
    vp.pan(-40, 25)
    p = Point(1234.5, 678.9)
    assert approx_pt(vp.to_image(vp.to_screen(p)), p)


def test_zoom_keeps_point_under_finger():
    vp = Viewport(1000, 1000, Rect(0, 0, 500, 500))
    anchor = Point(300, 200)
    before = vp.to_image(anchor)
    vp.zoom_at(4, anchor)
    assert vp.zoom == 4
    assert approx_pt(vp.to_image(anchor), before)


def test_zoom_limits_and_reset():
    vp = Viewport(100, 100, Rect(0, 0, 100, 100))
    vp.zoom_centre(1000)
    assert vp.zoom == Viewport.MAX_ZOOM
    vp.zoom_centre(1e-6)
    assert vp.zoom == 1
    vp.zoom_centre(3); vp.reset()
    assert (vp.zoom, vp.cx, vp.cy) == (1, 50, 50)


def test_pan_is_clamped_to_picture_edges():
    vp = Viewport(1000, 1000, Rect(0, 0, 500, 500))
    vp.pan(10_000, 0)                         # not zoomed: can't move at all
    assert vp.cx == 500
    vp.zoom_centre(2)                         # half the picture visible
    vp.pan(10_000, -10_000)
    r = vp.image_rect()
    assert r.x == pytest.approx(0) and r.y + r.h == pytest.approx(500)   # top-left corner pinned


def test_reading_is_identical_whether_tapped_zoomed_or_not():
    vp = Viewport(900, 1600, Rect(0, 0, 420, 780))
    s = Session(parts=100)
    s.base, s.top = Point(450, 1400), Point(450, 200)
    tap_img = Point(450, 1260.3)
    vp.zoom_at(5, vp.to_screen(tap_img))
    s.scale = vp.scale
    assert s.tap(vp.to_image(vp.to_screen(tap_img))) == "reading_added"
    assert s.reading_value(0) == pytest.approx(11.64, abs=0.01)


def test_hit_radius_stays_finger_sized_when_zoomed():
    s = Session(hit_radius=30)
    s.base, s.top = Point(0, 0), Point(0, 1000)
    assert s.target_at(Point(20, 0)) == "base"       # 20 px on screen at 1x
    s.scale = 4                                      # zoomed in: 20 picture px = 80 screen px
    assert s.target_at(Point(20, 0)) is None
    assert s.target_at(Point(5, 0)) == "base"


def test_grab_and_tap_are_separate():
    s = Session()
    assert s.tap(Point(0, 0)) == "base_set"
    assert s.grab(Point(1, 1)) == "base" and s.dragging == "base"
    s.drag(Point(0, 10)); s.release()
    assert s.base == Point(0, 10)
    assert s.grab(Point(300, 300)) is None
