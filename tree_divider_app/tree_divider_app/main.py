"""Entry point: `python main.py` on a computer, or packaged by Buildozer for Android."""
from kivy.config import Config

# Right-click is used for panning, so turn off Kivy's red multitouch-emulation dots.
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from tree_divider.ui.app import run  # noqa: E402

if __name__ == "__main__":
    run()
