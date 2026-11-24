"""
Command line: read the blue banner reading (e.g. 11.53) from Tree Divider pictures.

Usage:
    python tools/extract_reading.py picture.jpg [--parts 100] [--debug out.png] [--json]
    python tools/extract_reading.py my_folder/ --txt readings.txt       (many pictures)
    python tools/extract_reading.py my_folder/ --csv readings.csv

Tree height: the lowest blue dot is taken as the top of the banner (1 m unless
--banner-m says otherwise):   tree height = banner height x parts / banner reading

The machine vision itself lives in tree_divider/vision.py (also used by the app).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tree_divider import height  # noqa: E402
from tree_divider.vision import extract, parse_reading  # noqa: E402,F401  (re-exported for tests)


def _images(paths):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(exts):
                    yield os.path.join(p, name)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="picture files and/or folders")
    ap.add_argument("--parts", type=int, default=100, help="number of parts on the ruler (default 100)")
    ap.add_argument("--debug", help="save a picture showing what was detected (single picture)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--csv", help="write all readings to this CSV file (opens in Excel)")
    ap.add_argument("--txt", help="add each picture's readings and tree height to this text file "
                                  "(same format as the app's readings.txt)")
    ap.add_argument("--banner-m", type=float, default=height.DEFAULT_BANNER_M,
                    help="height of the banner in metres (default 1)")
    a = ap.parse_args(argv)
    if a.banner_m <= 0:
        ap.error("--banner-m must be greater than 0")

    rows, results, code = [], {}, 0
    for path in _images(a.images):
        try:
            res = extract(path, a.parts, a.debug)
        except Exception as exc:
            print(f"{os.path.basename(path)}: could not read ({exc})", file=sys.stderr)
            rows.append([os.path.basename(path), "", "", "", "", f"error: {exc}"])
            code = 1
            continue
        results[path] = asdict(res)
        values = [r.value for r in res.readings]          # lowest first = the banner
        h = height.compute(values, a.parts, a.banner_m)
        if h:
            results[path]["tree_height_m"] = round(h.tree_m, 3)
            print(f"{os.path.basename(path)}: tree height {h.tree_m:.2f} m "
                  f"({h.banner_m:g} m banner at {h.banner_reading:.2f} of {a.parts})")
        if a.txt:
            height.append_log(a.txt, height.log_line(os.path.basename(path), h, values, a.parts))
        if not res.readings:
            print(f"{os.path.basename(path)}: ruler found, but no blue reading dot")
            rows.append([os.path.basename(path), "", "", "", "", "no reading dot"])
        for r in res.readings:
            ocr = f"{r.ocr:.2f}" if r.ocr is not None else "not read"
            flag = "" if r.agree else "  <-- check: banner text and measurement differ" if r.agree is False else ""
            print(f"{os.path.basename(path)}:   reading {r.value:.2f}  [{r.source}; measured "
                  f"{r.geometric:.2f}, banner {ocr}]{flag}")
            rows.append([os.path.basename(path), f"{r.value:.2f}", r.source, f"{r.geometric:.2f}",
                         "" if r.ocr is None else f"{r.ocr:.2f}",
                         {True: "ok", False: "CHECK", None: "banner not read"}[r.agree]])
    if a.json:
        print(json.dumps(results, indent=2))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "reading", "source", "measured", "banner_text", "status"])
            # (tree heights are in the --txt output and the printed summary)
            w.writerows(rows)
        print(f"Saved {len(rows)} rows to {a.csv}")
    if a.txt:
        print(f"Readings added to {a.txt}")
    return code


if __name__ == "__main__":
    sys.exit(main())
