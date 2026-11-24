#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
[ -x venv/bin/python ] || { echo "Please run ./setup_mac_linux.sh first."; exit 1; }
exec venv/bin/python main.py
