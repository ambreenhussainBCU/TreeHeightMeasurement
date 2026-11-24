#!/usr/bin/env bash
# One-time setup on Mac or Linux: creates a private Python environment and installs everything.
cd "$(dirname "$0")" || exit 1
command -v python3 >/dev/null || { echo "Python 3 was not found. Install it from python.org (Mac) or your package manager (Linux)."; exit 1; }
python3 -m venv venv && . venv/bin/activate && python -m pip install --upgrade pip \
  && pip install -r requirements.txt -r requirements-vision.txt -r requirements-auto.txt \
  && echo && echo "Done! Start the app with:  ./run_mac_linux.sh"
