from datetime import datetime

import pytest

from tree_divider import height


def test_users_example_1m_banner_at_11_53():
    assert height.tree_height(11.53, 100, 1.0) == pytest.approx(8.673, abs=0.001)


def test_not_reading_divided_by_100():
    # 11.53 / 100 = 0.1153 would be the banner's share of the tree, not the tree's height
    assert height.tree_height(11.53, 100, 1.0) != pytest.approx(0.1153)


def test_other_points_and_banner_size():
    h = height.compute([10.0, 50.0, 100.0], 100, banner_m=1.5)
    assert h.tree_m == pytest.approx(15.0)
    assert h.points == [(50.0, pytest.approx(7.5)), (100.0, pytest.approx(15.0))]


def test_parts_setting_does_not_change_the_answer():
    # same physical picture measured with 10 parts instead of 100
    assert height.tree_height(1.153, 10, 1.0) == pytest.approx(height.tree_height(11.53, 100, 1.0))


def test_invalid_inputs():
    assert height.compute([], 100) is None
    assert height.compute([0.0], 100) is None
    with pytest.raises(ValueError):
        height.tree_height(5, 100, 0)


def test_log_file_has_header_once_and_one_line_per_save(tmp_path):
    path = str(tmp_path / "sub" / "readings.txt")
    when = datetime(2026, 10, 10, 12, 48, 48)
    h = height.compute([11.53, 45.2], 100, 1.0)
    height.append_log(path, height.log_line("a.jpg", h, [11.53, 45.2], 100, when))
    height.append_log(path, height.log_line("b.jpg", height.compute([20.0], 100), [20.0], 100, when))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 4                                   # title, header, 2 rows
    assert lines[1].split("\t")[6] == "Tree height (m)"
    row = lines[2].split("\t")
    assert row[:7] == ["2026-10-10", "12:48:48", "a.jpg", "100", "11.53", "1.00", "8.67"]
    assert row[7] == "45.20 = 3.92 m"
    assert lines[3].split("\t")[6] == "5.00"


def test_session_heights_first_dot_is_banner():
    from tree_divider.ruler import Point
    from tree_divider.session import Session
    s = Session(parts=100)
    s.tap(Point(0, 0)); s.tap(Point(0, 1000))
    assert "banner" in s.hint()
    s.tap(Point(0, 115.3)); s.tap(Point(0, 452))
    h = s.heights()
    assert h.tree_m == pytest.approx(8.673, abs=0.001)
    assert s.height_labels() == ["Banner 1 m", "3.92 m above base"]
    s.banner_m = 1.5
    assert s.heights().tree_m == pytest.approx(13.01, abs=0.01)


def test_saved_picture_caption_goes_in_a_strip_below():
    from PIL import Image
    from tree_divider import exporter
    from tree_divider.ruler import Point
    img = Image.new("RGB", (600, 800), (40, 90, 40))
    out = exporter.render(img, Point(300, 760), Point(300, 40), 100, [0.1153],
                          labels=["Banner 1 m"], caption="Tree height 8.67 m")
    assert out.width == 600 and out.height > 800                 # strip added
    assert out.getpixel((590, out.height - 3)) == exporter.INK   # dark strip
    assert out.getpixel((20, 20)) == (40, 90, 40)                # picture itself untouched there
