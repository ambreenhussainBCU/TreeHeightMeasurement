"""Zero-shot auto-measure: geometry and checks tested with stand-in models (fast, no download)."""
import os

import numpy as np
import pytest

pytest.importorskip("cv2")
from tree_divider import auto_measure as am  # noqa: E402


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, rgb, phrases, threshold):
        return self.dets


class FakeSegmenter:
    """Returns the drawn shapes: a tree (crown + trunk) and a banner rectangle."""
    def __init__(self, tree_mask, banner_mask):
        self.masks = {"tree": tree_mask, "banner": banner_mask}

    def set_image(self, rgb):
        pass

    def segment_box(self, box, prefer="score"):
        return self.masks["banner"].copy() if box == (180, 600, 210, 700) else self.masks["tree"].copy()


def scene(banner_top=600, banner_bottom=700, crown_top=100, base=700):
    h, w = 800, 400
    tree = np.zeros((h, w), bool)
    yy, xx = np.ogrid[:h, :w]
    tree |= ((xx - 150) ** 2 / 90 ** 2 + (yy - 250) ** 2 / 150 ** 2) <= 1     # crown, top at y=100
    tree[350:base + 1, 140:160] = True                                       # trunk to the base
    tree[:crown_top] = False
    banner = np.zeros((h, w), bool)
    banner[banner_top:banner_bottom + 1, 180:211] = True
    return tree, banner


def measurer(tree, banner, banner_box=(180, 600, 210, 700)):
    m = am.AutoMeasurer.__new__(am.AutoMeasurer)
    m.detector = FakeDetector([am.Detection("tree", 0.6, (60, 100, 240, 700)),
                               am.Detection("banner", 0.7, banner_box)])
    m.segmenter = FakeSegmenter(tree, banner)
    return m


def test_height_from_tree_and_banner_pixels():
    tree, banner = scene()
    r = measurer(tree, banner).measure(np.zeros((800, 400, 3), np.uint8), banner_m=1.0)
    assert r.tree_top[1] == 100 and r.tree_base[1] == 700
    assert r.banner_px == 100 and r.tree_px == 600
    assert r.tree_height_m == pytest.approx(6.0)
    assert r.warnings == []


def test_banner_size_setting():
    tree, banner = scene()
    r = measurer(tree, banner).measure(np.zeros((800, 400, 3), np.uint8), banner_m=1.5)
    assert r.tree_height_m == pytest.approx(9.0)


def test_warns_when_banner_not_at_base():
    tree, banner = scene(banner_top=450, banner_bottom=550)        # floating / farther away
    r = measurer(tree, banner).measure(np.zeros((800, 400, 3), np.uint8))
    assert any("base" in w for w in r.warnings)


def test_warns_when_banner_held_near_camera():
    tree, banner = scene(banner_top=150, banner_bottom=700)
    r = measurer(tree, banner).measure(np.zeros((800, 400, 3), np.uint8))
    assert any("close" in w for w in r.warnings)


def test_reports_missing_banner():
    tree, banner = scene()
    m = measurer(tree, banner)
    m.detector = FakeDetector([am.Detection("tree", 0.6, (60, 100, 240, 700))])
    r = m.measure(np.zeros((800, 400, 3), np.uint8))
    assert r.tree_height_m is None and "No banner found." in r.warnings


def test_nms_keeps_best_of_overlapping_boxes():
    d = [am.Detection("tree", 0.4, (0, 0, 100, 100)), am.Detection("tree", 0.6, (5, 5, 100, 100)),
         am.Detection("banner", 0.5, (0, 0, 100, 100))]
    kept = am._nms(d)
    assert sorted((k.label, k.score) for k in kept) == [("banner", 0.5), ("tree", 0.6)]


MODELS = am.DEFAULT_MODEL_DIR
HAS_TOKENIZER = os.path.exists(os.path.join(MODELS, "bert_base_uncased_tokenizer.json"))


@pytest.mark.skipif(not HAS_TOKENIZER, reason="models not downloaded")
def test_text_encoding_keeps_phrases_separate():
    from tokenizers import Tokenizer
    gd = am.GroundingDINO.__new__(am.GroundingDINO)
    gd.tokenizer = Tokenizer.from_file(os.path.join(MODELS, "bert_base_uncased_tokenizer.json"))
    feeds, spans = gd._encode_text(["tree", "white board"])
    ids = feeds["input_ids"][0].tolist()
    words = [gd.tokenizer.id_to_token(ids[i]) for a, b in spans for i in range(a, b)]
    assert words == ["tree", "white", "board"]
    m = feeds["text_token_mask"][0]
    a, b = spans[0]
    c, d = spans[1]
    assert m[a, a] and not m[a, c]                 # "tree" doesn't attend to "white board"
    assert feeds["position_ids"][0, c] == 0        # positions restart for each phrase


def test_pole_next_to_tree_is_ignored_for_the_base():
    h, w = 800, 400
    mask = np.zeros((h, w), bool)
    yy, xx = np.ogrid[:h, :w]
    mask |= ((xx - 150) ** 2 / 110 ** 2 + (yy - 250) ** 2 / 150 ** 2) <= 1   # crown, top y=100
    mask[380:701, 142:160] = True                                          # trunk, ground at y=700
    mask[300:761, 252:262] = True                                          # pole touching the crown, lower
    clean, top, base = am.tree_points(mask)
    assert top[1] == 100
    assert base[1] == 700 and abs(base[0] - 150) < 6                       # the trunk, not the pole
    assert not clean[600:761, 252:262].any()                               # pole removed below the crown


def test_banner_ignores_thin_stray_bits():
    mask = np.zeros((800, 400), bool)
    mask[600:701, 180:211] = True          # the banner, 100 px tall
    mask[560:600, 195:197] = True          # a thin stray bit above it (e.g. background edge)
    top, bottom = am.banner_points(mask)
    assert top[1] == 600 and bottom[1] == 700


class BoxSegmenter:
    """Stand-in outliner: returns a prepared mask for each box."""
    def __init__(self, masks):
        self.masks = masks

    def set_image(self, rgb):
        pass

    def segment_box(self, box, prefer="score"):
        return self.masks[tuple(box)].copy()


def tree_pole_scene():
    h, w = 800, 400
    yy, xx = np.ogrid[:h, :w]
    tree = np.zeros((h, w), bool)
    tree |= ((xx - 150) ** 2 / 110 ** 2 + (yy - 250) ** 2 / 150 ** 2) <= 1   # crown, top y=100
    tree[380:701, 142:160] = True                                           # trunk to y=700
    pole = np.zeros((h, w), bool)
    pole[150:741, 250:258] = True                                           # pole, taller & lower
    banner = np.zeros((h, w), bool)
    banner[600:701, 180:211] = True
    return tree, pole, banner


def test_pole_joined_to_trunk_at_ground_is_still_separated():
    tree, pole, _ = tree_pole_scene()
    mask = tree | pole
    mask[690:701, 140:260] = True                     # kerb/shadow joining trunk and pole
    clean, top, base = am.tree_points(mask)
    assert abs(base[0] - 150) < 8 and base[1] == 700  # trunk, not the pole (which ends at 740)
    assert not clean[600:741, 250:258].any()


def _measurer(dets, masks):
    m = am.AutoMeasurer.__new__(am.AutoMeasurer)
    m.detector = FakeDetector(dets)
    m.segmenter = BoxSegmenter(masks)
    return m


def test_pole_beside_tree_is_cut_out_of_the_outline():
    tree, pole, banner = tree_pole_scene()
    tbox, pbox, bbox = (30, 95, 270, 745), (245, 145, 262, 745), (180, 600, 210, 700)
    m = _measurer([am.Detection("tree", 0.6, tbox), am.Detection("pole", 0.5, pbox),
                   am.Detection("banner", 0.7, bbox)],
                  {tbox: tree | pole, pbox: pole, bbox: banner})
    r = m.measure(np.zeros((800, 400, 3), np.uint8))
    assert not r.tree_mask[150:741, 250:258].any()     # pole gone from the tree's outline
    assert r.tree_base[1] == 700 and r.tree_top[1] == 100
    assert r.tree_height_m == pytest.approx(6.0)


def test_only_a_pole_is_not_called_a_tree():
    _, pole, banner = tree_pole_scene()
    pbox, bbox = (245, 145, 262, 745), (180, 600, 210, 700)
    m = _measurer([am.Detection("pole", 0.6, pbox), am.Detection("banner", 0.7, bbox)],
                  {pbox: pole, bbox: banner})
    r = m.measure(np.zeros((800, 400, 3), np.uint8))
    assert r.tree_height_m is None
    assert any("No tree found (found: pole)" in w for w in r.warnings)


def test_measuring_a_pole_on_purpose():
    _, pole, banner = tree_pole_scene()
    banner2 = np.zeros_like(banner)
    banner2[640:741, 270:300] = True                  # banner at the pole's base (y=740)
    pbox, bbox = (245, 145, 262, 745), (270, 640, 300, 740)
    m = _measurer([am.Detection("pole", 0.6, pbox), am.Detection("banner", 0.7, bbox)],
                  {pbox: pole, bbox: banner2})
    r = m.measure(np.zeros((800, 400, 3), np.uint8), tree_words=("pole",))
    assert r.tree_top[1] == 150 and r.tree_base[1] == 740
    assert r.tree_height_m == pytest.approx(5.9)


def test_tree_label_on_a_more_confident_pole_is_ignored():
    _, pole, banner = tree_pole_scene()
    pbox, bbox = (245, 145, 262, 745), (180, 600, 210, 700)
    m = _measurer([am.Detection("pole", 0.6, pbox), am.Detection("tree", 0.4, (244, 146, 263, 744)),
                   am.Detection("banner", 0.7, bbox)],
                  {pbox: pole, (244, 146, 263, 744): pole, bbox: banner})
    r = m.measure(np.zeros((800, 400, 3), np.uint8))
    assert r.tree_height_m is None and any("found: pole" in w for w in r.warnings)


def test_chosen_pole_is_measured_even_if_it_looks_more_like_a_tree():
    _, pole, banner = tree_pole_scene()
    banner2 = np.zeros_like(banner)
    banner2[640:741, 270:300] = True
    pbox, tbox, bbox = (245, 145, 262, 745), (244, 146, 263, 744), (270, 640, 300, 740)
    m = _measurer([am.Detection("tree", 0.46, tbox), am.Detection("pole", 0.35, pbox),
                   am.Detection("banner", 0.7, bbox)],
                  {pbox: pole, tbox: pole, bbox: banner2})
    r = m.measure(np.zeros((800, 400, 3), np.uint8), tree_words=("pole",))
    assert r.tree_height_m == pytest.approx(5.9)


def test_banner_nearest_the_base_wins_over_a_higher_score_elsewhere():
    tree, _, banner = tree_pole_scene()
    door = np.zeros_like(banner)
    door[400:560, 20:60] = True                                   # white-framed door, far away
    tbox, bbox, dbox = (30, 95, 270, 705), (180, 600, 210, 700), (20, 400, 60, 560)
    m = _measurer([am.Detection("tree", 0.6, tbox), am.Detection("banner", 0.30, dbox),
                   am.Detection("banner", 0.26, bbox)],
                  {tbox: tree, bbox: banner, dbox: door})
    r = m.measure(np.zeros((800, 400, 3), np.uint8))
    assert r.banner_box == bbox and r.tree_height_m == pytest.approx(6.0)
