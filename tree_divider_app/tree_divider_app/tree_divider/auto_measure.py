"""
Zero-shot automatic tree measurement (prototype).

Upload a photo of a whole tree with a banner of known height standing at its base;
no tapping and no training:

  1. Grounding DINO finds the "tree" and the "banner" from those words alone
     (open-vocabulary detection - it was never trained on your trees).
  2. SAM 2.1 outlines each one precisely inside its box (segmentation).
  3. From the outlines:  tree top = highest tree pixel,  tree base = lowest trunk
     pixel,  banner = top and bottom of its outline.
  4. tree height = banner height x (tree pixels / banner pixels)

Both models run with onnxruntime on an ordinary CPU (no graphics card, no PyTorch).
They are downloaded automatically on first use (~335 MB) into ~/.tree_divider/models.

Models: Grounding DINO (IDEA Research, Apache-2.0) and SAM 2.1 (Meta, Apache-2.0),
in the ONNX conversions published on GitHub by the X-AnyLabeling project.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------- model files
_RELEASES = "https://github.com/CVHub520/X-AnyLabeling/releases/download"
_RAW = "https://raw.githubusercontent.com/CVHub520/X-AnyLabeling/main"
MODEL_FILES = {
    "groundingdino_swint_ogc_quant.onnx": f"{_RELEASES}/v1.0.0/groundingdino_swint_ogc_quant.onnx",
    "sam2.1_hiera_tiny.encoder.onnx": f"{_RELEASES}/v2.5.0/sam2.1_hiera_tiny.encoder.onnx",
    "sam2.1_hiera_tiny.decoder.onnx": f"{_RELEASES}/v2.5.0/sam2.1_hiera_tiny.decoder.onnx",
    "bert_base_uncased_tokenizer.json":
        f"{_RAW}/anylabeling/services/auto_labeling/configs/bert/bert_base_uncased_tokenizer.json",
}
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".tree_divider", "models")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_models(model_dir: str = DEFAULT_MODEL_DIR, progress=print) -> str:
    """Download any missing model files. Returns the folder."""
    os.makedirs(model_dir, exist_ok=True)
    for name, url in MODEL_FILES.items():
        path = os.path.join(model_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        if progress:
            progress(f"Downloading {name} (first run only)...")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return model_dir


def _session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- Grounding DINO
@dataclass
class Detection:
    label: str            # which prompt it matched, e.g. "tree"
    score: float
    box: tuple            # x1, y1, x2, y2 in picture pixels


# Things easily confused with what's being measured, looked for so they can be told apart.
DISTRACTORS = {"tree": ("pole", "lamp post"), "pole": ("tree",), "lamp post": ("tree",)}


class GroundingDINO:
    """Open-vocabulary detector: finds things described in words."""

    INPUT_W, INPUT_H = 1200, 800
    SEPARATORS = (101, 102, 1012, 1029)       # [CLS] [SEP] "." "?"

    def __init__(self, model_dir: str):
        from tokenizers import Tokenizer
        self.session = _session(os.path.join(model_dir, "groundingdino_swint_ogc_quant.onnx"))
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "bert_base_uncased_tokenizer.json"))

    def _encode_text(self, phrases: Sequence[str]):
        caption = " . ".join(p.lower().strip() for p in phrases) + " ."
        enc = self.tokenizer.encode(caption)
        ids = np.array([enc.ids], dtype=np.int64)
        n = ids.shape[1]
        # Each phrase only attends to its own words, and word positions restart per
        # phrase (the "sub-sentence" encoding the model was trained with).
        attn = np.zeros((1, n, n), dtype=bool)
        pos = np.zeros((1, n), dtype=np.int64)
        seps = [i for i, t in enumerate(enc.ids) if t in self.SEPARATORS]
        spans, prev = [], 0
        for s in seps:
            if s == 0 or s == n - 1:
                attn[0, s, s] = True
            else:
                attn[0, prev + 1:s + 1, prev + 1:s + 1] = True
                pos[0, prev + 1:s + 1] = np.arange(0, s - prev)
                spans.append((prev + 1, s))          # tokens of one phrase (without the ".")
            prev = s
        feeds = {
            "input_ids": ids,
            "attention_mask": np.array([enc.attention_mask], dtype=bool),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
            "position_ids": pos,
            "text_token_mask": attn,
        }
        return feeds, spans[:len(phrases)]

    def detect(self, rgb: np.ndarray, phrases: Sequence[str], threshold: float = 0.3) -> List[Detection]:
        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        feeds, spans = self._encode_text(phrases)
        feeds["img"] = img.transpose(2, 0, 1)[None].astype(np.float32)
        logits, boxes = self.session.run(None, feeds)
        probs = 1 / (1 + np.exp(-logits[0]))          # (queries, tokens)
        boxes = boxes[0]                              # (queries, 4) centre-x, centre-y, w, h in 0..1
        # score of each query for each phrase = its strongest word in that phrase
        scores = np.stack([probs[:, a:b].max(axis=1) for a, b in spans], axis=1)
        out = []
        for q in np.nonzero(scores.max(axis=1) > threshold)[0]:
            k = int(scores[q].argmax())
            cx, cy, bw, bh = boxes[q] * np.array([w, h, w, h])
            out.append(Detection(phrases[k], float(scores[q, k]),
                                 (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
        return _nms(out)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def _nms(dets: List[Detection], iou: float = 0.6) -> List[Detection]:
    keep = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.label != d.label or _iou(k.box, d.box) < iou for k in keep):
            keep.append(d)
    return keep


# ---------------------------------------------------------------- SAM 2.1
class SAM2:
    """Segment Anything 2.1: precise outline of the object inside a box."""

    SIZE = 1024

    def __init__(self, model_dir: str):
        self.encoder = _session(os.path.join(model_dir, "sam2.1_hiera_tiny.encoder.onnx"))
        self.decoder = _session(os.path.join(model_dir, "sam2.1_hiera_tiny.decoder.onnx"))
        self._feats = None
        self._size = None

    def set_image(self, rgb: np.ndarray):
        self._size = rgb.shape[:2]
        self._lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        img = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        self._feats = self.encoder.run(None, {"image": img.transpose(2, 0, 1)[None].astype(np.float32)})

    def segment_box(self, box, prefer: str = "score") -> np.ndarray:
        """Boolean mask (picture size) of the main object in `box`.

        SAM proposes a few candidate outlines. prefer="score" takes the one SAM is most
        confident about; prefer="box" takes the one that best fills the box;
        prefer="uniform" (for a banner) takes, among outlines at least 60% as tall as the
        box, the most evenly coloured one. A plain board outlined on its own is uniform;
        an outline that has spilled into a car or wall behind it isn't. (Tested on real
        photos: the box-filling choice spilled into a silver car and grass behind white
        boards, making them 10-15% too tall.)
        """
        h, w = self._size
        sx, sy = self.SIZE / w, self.SIZE / h
        x1, y1, x2, y2 = box
        coords = np.array([[[x1 * sx, y1 * sy], [x2 * sx, y2 * sy]]], dtype=np.float32)
        labels = np.array([[2, 3]], dtype=np.float32)            # 2,3 = box corners
        hr0, hr1, embed = self._feats
        masks, ious = self.decoder.run(None, {
            "image_embed": embed, "high_res_feats_0": hr0, "high_res_feats_1": hr1,
            "point_coords": coords, "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        })
        cands = [cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0 for m in masks[0]]
        if prefer == "uniform" and getattr(self, "_lab", None) is not None:
            box_h = max(1.0, y2 - y1)
            def height(m):
                rows = np.nonzero(m.any(axis=1))[0]
                return rows.max() - rows.min() + 1 if len(rows) else 0
            ok = [i for i, m in enumerate(cands) if m.sum() > 20 and height(m) >= 0.6 * box_h]
            ok = ok or [i for i, m in enumerate(cands) if m.sum() > 20] or [int(np.argmax(ious[0]))]
            spread = {i: float(self._lab[cands[i]].std(axis=0).mean()) for i in ok}
            return cands[min(ok, key=lambda i: spread[i])]
        if prefer == "box":
            rect = np.zeros((h, w), bool)
            rect[max(0, int(y1)):int(np.ceil(y2)), max(0, int(x1)):int(np.ceil(x2))] = True
            fill = [(m & rect).sum() / max(1, (m | rect).sum()) for m in cands]
            return cands[int(np.argmax(fill))]
        return cands[int(np.argmax(ious[0]))]


# ---------------------------------------------------------------- measurement
@dataclass
class AutoResult:
    tree_height_m: Optional[float]
    banner_m: float
    tree_top: Optional[tuple] = None        # (x, y) pixels
    tree_base: Optional[tuple] = None
    banner_top: Optional[tuple] = None
    banner_bottom: Optional[tuple] = None
    tree_px: float = 0.0
    banner_px: float = 0.0
    tree_score: float = 0.0
    banner_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    tree_box: Optional[tuple] = None        # the detection chosen as the tree (x1, y1, x2, y2)
    banner_box: Optional[tuple] = None
    tree_mask: Optional[np.ndarray] = field(default=None, repr=False)     # exact outlines
    banner_mask: Optional[np.ndarray] = field(default=None, repr=False)


def _extremes(mask: np.ndarray):
    """Top point and bottom point of a mask (bottom = centre of its lowest rows)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None, None
    top_y = ys.min()
    top = (float(xs[ys == top_y].mean()), float(top_y))
    bot_y = ys.max()
    band = ys >= bot_y - max(2, int(0.01 * (bot_y - top_y)))
    bottom = (float(np.median(xs[band])), float(bot_y))
    return top, bottom


def _tree_base(mask: np.ndarray, top_y: int) -> tuple:
    """Lowest point of the trunk (kept for compatibility; see tree_points)."""
    return tree_points(mask)[2]


def tree_points(mask: np.ndarray):
    """Clean a tree outline and return (clean_mask, top, base).

    Below the crown, the outline can also contain things standing next to the tree
    (lamp posts, poles, fences) because they share its box. Only the stem under the
    crown's centre is kept; the base is the lowest point of that stem.
    """
    mask = mask.copy()
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask, None, None
    top_y = int(ys.min())
    top = (float(xs[ys == top_y].mean()), float(top_y))
    widths = mask.sum(axis=1)
    widest = int(np.argmax(widths))
    # crown bottom: first row below the widest row where the outline gets narrow
    narrow = np.nonzero(widths[widest:] < 0.25 * widths[widest])[0]
    crown_bottom = widest + int(narrow[0]) if len(narrow) else int(ys.max())
    crown = mask[top_y:crown_bottom + 1]
    cys, cxs = np.nonzero(crown)
    centre_x = float(np.median(cxs)) if len(cxs) else float(np.median(xs))
    lower = mask[crown_bottom:]
    if lower.any():
        # Below the crown, find the separate tall vertical things (trunk, poles, posts) by
        # their columns, so they're told apart even if joined at the ground by a kerb,
        # shadow or low wall. Keep the one nearest the crown's centre: the trunk.
        cols = lower.sum(axis=0)
        tall = cols >= max(3, 0.3 * cols.max())
        runs, start_x, gap = [], None, 0
        for x in range(len(tall) + 1):
            if x < len(tall) and tall[x]:
                if start_x is None:
                    start_x = x
                gap, last = 0, x
            elif start_x is not None:
                gap += 1
                if gap > 2 or x == len(tall):
                    runs.append((start_x, last))
                    start_x, gap = None, 0
        if runs:
            def score(run):
                return (abs((run[0] + run[1]) / 2 - centre_x), -int(cols[run[0]:run[1] + 1].sum()))
            x1, x2 = min(runs, key=score)
            keep = np.zeros_like(lower)
            keep[:, x1:x2 + 1] = lower[:, x1:x2 + 1]
            mask[crown_bottom:] = keep
            sy, sx = np.nonzero(keep)
            base_y = int(sy.max())
            base_x = float(np.median(sx[sy >= base_y - max(2, int(0.01 * (base_y - top_y)))]))
            return mask, top, (base_x, float(base_y + crown_bottom))
    base_y = int(ys.max())
    return mask, top, (float(np.median(xs[ys >= base_y - 2])), float(base_y))


def stem_columns(mask: np.ndarray, base) -> tuple:
    """Column range of the stem just above the base (to tell the trunk from a pole)."""
    if base is None:
        return None
    by = int(base[1])
    ys, xs = np.nonzero(mask[max(0, by - 40):by + 1])
    return (int(xs.min()), int(xs.max())) if len(xs) else None


def banner_points(mask: np.ndarray):
    """Top and bottom of a banner from its solid, rectangular part (ignoring thin bits)."""
    widths = mask.sum(axis=1)
    rows = np.nonzero(widths)[0]
    if len(rows) == 0:
        return None, None
    solid = rows[widths[rows] >= 0.5 * np.median(widths[rows])]
    top_y, bot_y = int(solid.min()), int(solid.max())
    xs_top = np.nonzero(mask[top_y])[0]
    xs_bot = np.nonzero(mask[bot_y])[0]
    return (float(xs_top.mean()), float(top_y)), (float(xs_bot.mean()), float(bot_y))


def choose_banner(banners: List[Detection], top, base) -> Detection:
    """The banner-like detection standing nearest the object's base (score breaks ties)."""
    if base is None or top is None:
        return max(banners, key=lambda d: d.score)
    height = max(1.0, base[1] - top[1])

    def cost(d):
        bx, by = (d.box[0] + d.box[2]) / 2, d.box[3]          # banner's bottom centre
        dist = float(np.hypot(bx - base[0], by - base[1])) / height
        return dist - 0.5 * d.score
    return min(banners, key=cost)


class AutoMeasurer:
    def __init__(self, model_dir: Optional[str] = None, progress=print):
        model_dir = ensure_models(model_dir or DEFAULT_MODEL_DIR, progress)
        self.detector = GroundingDINO(model_dir)
        self.segmenter = SAM2(model_dir)

    def measure(self, rgb: np.ndarray, banner_m: float = 1.0,
                tree_words: Sequence[str] = ("tree",),
                banner_words: Sequence[str] = ("banner",),
                threshold: float = 0.25,
                distractor_words: Optional[Sequence[str]] = None) -> AutoResult:
        """Measure the object described by `tree_words` (a tree by default).

        `distractor_words` are things to tell apart from it (for a tree: poles and lamp
        posts). The detector labels each object with the best-matching words, so a pole
        is labelled "pole" instead of being taken for the tree, and is cut out of the
        tree's outline if it stands beside it.
        """
        res = AutoResult(None, banner_m)
        target = tree_words[0]
        if distractor_words is None:
            distractor_words = [w for t in tree_words for w in DISTRACTORS.get(t.strip().lower(), ())]
        distractor_words = [w for w in distractor_words
                            if w not in tree_words and w not in banner_words]
        dets = self.detector.detect(rgb, list(tree_words) + list(banner_words) + distractor_words,
                                    threshold)
        res.detections = dets
        others = [d for d in dets if d.label in distractor_words]
        # When measuring a tree: a "tree" box sitting on the same object as a more confident
        # "pole" box is that pole. (Not applied when a pole is chosen on purpose: a wooden
        # telegraph pole can look more like a tree to the detector than a pole.)
        suppress = target.strip().lower() == "tree"
        trees = [d for d in dets if d.label in tree_words
                 and not (suppress and any(_iou(d.box, o.box) > 0.5 and o.score >= d.score for o in others))]
        banners = [d for d in dets if d.label in banner_words]
        if not trees:
            found = sorted({d.label for d in others})
            res.warnings.append(f"No {target} found" + (f" (found: {', '.join(found)})." if found else ".")
                                + (" To measure that instead, change what to measure." if found else ""))
        if not banners:
            res.warnings.append("No banner found.")
        if not trees or not banners:
            return res

        # The tree: the tallest confident detection.
        tree = max(trees, key=lambda d: (d.box[3] - d.box[1]) * d.score)
        self.segmenter.set_image(rgb)
        raw = self.segmenter.segment_box(tree.box)
        _, top0, base0 = tree_points(raw)
        # The banner: must stand at the base, so take the banner-like thing nearest the
        # base, not simply the highest score (a white-framed door or window can score
        # higher than the real banner).
        banner = choose_banner(banners, top0, base0)
        res.tree_score, res.banner_score = tree.score, banner.score
        res.tree_box, res.banner_box = tree.box, banner.box
        bmask = self.segmenter.segment_box(banner.box, prefer="uniform")
        raw &= ~bmask                                    # the banner isn't part of the tree
        tmask, t_top, t_base = tree_points(raw)
        # Poles / posts standing beside the tree: cut their outlines out of the tree's.
        stem = stem_columns(tmask, t_base)
        removed = []
        for d in others:
            ix = max(0.0, min(d.box[2], tree.box[2]) - max(d.box[0], tree.box[0]))
            iy = max(0.0, min(d.box[3], tree.box[3]) - max(d.box[1], tree.box[1]))
            area = max(1.0, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            if ix * iy / area < 0.3:
                continue                                 # not inside the tree's box
            dmask = self.segmenter.segment_box(d.box)
            dxs = np.nonzero(dmask.any(axis=0))[0]
            if stem and len(dxs):
                overlap = max(0, min(stem[1], dxs.max()) - max(stem[0], dxs.min()) + 1)
                if overlap > 0.5 * (stem[1] - stem[0] + 1):
                    continue                             # that "pole" is really the trunk: keep it
            raw &= ~dmask
            removed.append(d.label)
        if removed:
            tmask, t_top, t_base = tree_points(raw)
        res.tree_mask, res.banner_mask = tmask, bmask   # the cleaned outlines (what is measured)
        b_top, b_bot = banner_points(bmask)
        if t_top is None or b_top is None:
            res.warnings.append(f"Couldn't outline the {target} or the banner.")
            return res

        res.tree_top, res.tree_base = t_top, t_base
        res.banner_top, res.banner_bottom = b_top, b_bot
        res.tree_px = t_base[1] - t_top[1]
        res.banner_px = b_bot[1] - b_top[1]
        if res.banner_px <= 2 or res.tree_px <= 2:
            res.warnings.append("Tree or banner too small to measure.")
            return res
        res.tree_height_m = banner_m * res.tree_px / res.banner_px

        # ---- sanity checks: when the photo breaks the method's assumptions
        gap = abs(b_bot[1] - t_base[1]) / res.tree_px
        if gap > 0.08:
            res.warnings.append(
                f"The banner's bottom is {gap:.0%} of the tree height away from the tree's base. "
                "It should stand at the base, the same distance from the camera as the trunk; "
                "otherwise the height is wrong.")
        if res.banner_px > 0.6 * res.tree_px:
            res.warnings.append("The banner looks very large next to the tree: is it held close "
                                "to the camera? That makes the tree height far too low.")
        h = rgb.shape[0]
        if t_top[1] <= 2:
            res.warnings.append("The treetop touches the top edge of the photo: the top may be cut off.")
        if t_base[1] >= h - 3:
            res.warnings.append("The tree reaches the bottom edge of the photo: the base may be cut off.")
        if tree.score < 0.35 or banner.score < 0.35:
            res.warnings.append("Low detection confidence: check the marked picture.")
        return res


# ---------------------------------------------------------------- picture of the result
def draw_result(rgb: np.ndarray, res: AutoResult, tree_mask=None, banner_mask=None,
                name: str = "tree") -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s = max(out.shape[:2]) / 800
    tree_mask = res.tree_mask if tree_mask is None else tree_mask
    banner_mask = res.banner_mask if banner_mask is None else banner_mask
    ORANGE = (0, 140, 255)                                    # (OpenCV colours are blue-green-red)
    for mask, colour in ((tree_mask, (60, 200, 60)), (banner_mask, ORANGE)):
        if mask is not None and mask.any():                  # tinted fill + outline
            tint = out.copy()
            tint[mask] = colour
            out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colour, max(1, int(2 * s)))
    for i, d in enumerate(res.detections):
        chosen = d.box in (res.tree_box, res.banner_box)
        # green = what was measured, orange = the banner used, grey = found but ignored
        colour = ((60, 200, 60) if d.box == res.tree_box else ORANGE if d.box == res.banner_box
                  else (150, 150, 150))
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, max(1, int((3 if chosen else 1) * s)))
        label = f"#{i + 1} {d.label} {d.score:.2f}"
        org = (x1 + 4, max(int(16 * s), y1 + int(20 * s)))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), max(3, int(4 * s)))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, colour, max(1, int(1.5 * s)))

    def mark(p, colour, text):
        if p:
            c = (int(p[0]), int(p[1]))
            cv2.circle(out, c, int(8 * s), (255, 255, 255), -1)
            cv2.circle(out, c, int(6 * s), colour, -1)
            cv2.putText(out, text, (c[0] + int(12 * s), c[1] + int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * s, (255, 255, 255), max(2, int(3 * s)))
            cv2.putText(out, text, (c[0] + int(12 * s), c[1] + int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * s, colour, max(1, int(1.5 * s)))

    mark(res.tree_top, (139, 61, 255), f"{name} top")
    mark(res.tree_base, (63, 210, 255), f"{name} base")
    mark(res.banner_top, (255, 160, 41), "banner top")
    mark(res.banner_bottom, (255, 160, 41), "banner bottom")
    text = (f"{name.capitalize()} height {res.tree_height_m:.2f} m" if res.tree_height_m else "No measurement")
    pad = int(10 * s)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s)))
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (18, 23, 27), -1)
    cv2.putText(out, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (63, 210, 255),
                max(1, int(2 * s)))
    return out
