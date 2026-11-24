#!/usr/bin/env bash
# Measure tree photos automatically:  ./auto_measure_mac_linux.sh photo1.jpg photo2.jpg  (or a folder)
BANNER_M=1
BANNER_WORDS="banner"      # describe your banner, e.g. "white board", "orange board"
cd "$(dirname "$0")" || exit 1
[ -x venv/bin/python ] || { echo "Please run ./setup_mac_linux.sh first."; exit 1; }
[ $# -gt 0 ] || { echo "Usage: $0 photo.jpg [more photos or folders]"; exit 1; }
OUT="$HOME/Pictures/TreeDivider/Auto"
echo "Results will go to: $OUT   (first run downloads ~335 MB of models)"
venv/bin/python tools/auto_measure.py "$@" --banner-m "$BANNER_M" --banner-words "$BANNER_WORDS" \
  --out "$OUT" --txt "$OUT/auto-readings.txt"
