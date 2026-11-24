"""
Automatic tree height from a photo: no tapping, no training (zero-shot prototype).

Photograph the whole tree with a banner of known height standing upright at its base.

Usage:
    python tools/auto_measure.py photo.jpg
    python tools/auto_measure.py photos/ --out results/ --txt auto-readings.txt
    python tools/auto_measure.py photo.jpg --banner-m 1.5 --banner-words "white board"

Options:
    --banner-m       banner height in metres (default 1)
    --banner-words   how to describe the banner to the detector (default "banner");
                     try "white board", "sign", "poster", "orange board"...
    --tree-words     how to describe the tree (default "tree")
    --out            folder for marked pictures showing what was found
    --txt            add one line per photo to this text file

First run downloads the models (~335 MB) into ~/.tree_divider/models.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tree_divider.auto_measure import AutoMeasurer, draw_result  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
COLUMNS = ["Date", "Time", "Picture", "Tree height (m)", "Banner height (m)", "Tree (px)",
           "Banner (px)", "Tree score", "Banner score", "Warnings"]


def images(paths):
    for p in paths:
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                if n.lower().endswith(EXTS):
                    yield os.path.join(p, n)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--banner-m", type=float, default=1.0)
    ap.add_argument("--banner-words", default="banner")
    ap.add_argument("--tree-words", default="tree")
    ap.add_argument("--out", help="folder for marked pictures")
    ap.add_argument("--txt", help="text file to add results to")
    ap.add_argument("--models", help="model folder (default ~/.tree_divider/models)")
    a = ap.parse_args(argv)
    if a.banner_m <= 0:
        ap.error("--banner-m must be greater than 0")

    print("Loading models...")
    m = AutoMeasurer(a.models)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    code = 0
    for path in images(a.images):
        bgr = cv2.imread(path)
        name = os.path.basename(path)
        if bgr is None:
            print(f"{name}: can't open")
            code = 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = m.measure(rgb, a.banner_m, tree_words=(a.tree_words,), banner_words=(a.banner_words,))
        if r.tree_height_m:
            print(f"{name}: tree height {r.tree_height_m:.2f} m  "
                  f"(tree {r.tree_px:.0f} px, {a.banner_m:g} m banner {r.banner_px:.0f} px)")
        else:
            print(f"{name}: no measurement")
            code = 1
        for w in r.warnings:
            print(f"   ! {w}")
        if a.out:
            out = os.path.join(a.out, os.path.splitext(name)[0] + "-auto.jpg")
            cv2.imwrite(out, draw_result(rgb, r))
        if a.txt:
            new = not os.path.exists(a.txt) or os.path.getsize(a.txt) == 0
            now = datetime.now()
            with open(a.txt, "a", encoding="utf-8") as f:
                if new:
                    f.write("Tree Divider automatic measurements (tab-separated)\n")
                    f.write("\t".join(COLUMNS) + "\n")
                f.write("\t".join([
                    now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name,
                    f"{r.tree_height_m:.2f}" if r.tree_height_m else "",
                    f"{a.banner_m:.2f}", f"{r.tree_px:.0f}", f"{r.banner_px:.0f}",
                    f"{r.tree_score:.2f}", f"{r.banner_score:.2f}", " | ".join(r.warnings),
                ]) + "\n")
    if a.txt:
        print(f"Results added to {a.txt}")
    return code


if __name__ == "__main__":
    sys.exit(main())
