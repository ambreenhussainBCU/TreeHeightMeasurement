"""
Choosing a picture.

Order tried:
  Android  -> the phone's own picker / Gallery (plyer)
  Windows / Linux -> the system "Open" window (tkinter), then plyer
  Mac      -> plyer, then tkinter
  Anything fails -> PhotoBrowser, a built-in browser with folder shortcuts and a preview.

`pick_image(callback)` always calls `callback(path)` or `callback(None)` on the
Kivy main thread.
"""
from __future__ import annotations

import os

from kivy.clock import Clock
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.utils import platform

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")
_patterns = [f"*{e}" for e in IMAGE_EXTS] + [f"*{e.upper()}" for e in IMAGE_EXTS]


def _later(callback, value):
    Clock.schedule_once(lambda *_: callback(value), 0)


def start_folder() -> str:
    home = os.path.expanduser("~")
    for sub in ("Pictures", "DCIM", "Downloads", "Desktop"):
        path = os.path.join(home, sub)
        if os.path.isdir(path):
            return path
    return home


# ---- pickers ------------------------------------------------------------------
def _tk_picker(callback) -> bool:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            parent=root,
            title="Choose a photo of a tree",
            initialdir=start_folder(),
            filetypes=[("Pictures", " ".join(_patterns)), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    _later(callback, path or None)
    return True


def _plyer_picker(callback) -> bool:
    from plyer import filechooser

    def chosen(selection):
        _later(callback, selection[0] if selection else None)

    if platform == "android":
        filechooser.open_file(on_selection=chosen, filters=["image"])
    else:
        filechooser.open_file(on_selection=chosen, path=start_folder(),
                              title="Choose a photo of a tree",
                              filters=[["Pictures"] + _patterns])
    return True


def pick_image(callback) -> None:
    if platform == "android":
        order = (_plyer_picker,)
    elif platform == "macosx":
        order = (_plyer_picker, _tk_picker)
    else:
        order = (_tk_picker, _plyer_picker)
    for picker in order:
        try:
            if picker(callback):
                return
        except Exception as exc:          # missing library, no dialog tool installed, etc.
            Logger.info(f"TreeDivider: {picker.__name__} unavailable ({exc}); trying next")
    PhotoBrowser(callback).open()


# ---- built-in browser (fallback) --------------------------------------------------
class PhotoBrowser(Popup):
    """Folder shortcuts, big icons, a live preview and a clear Open button."""

    def __init__(self, callback, **kwargs):
        super().__init__(title="Choose a photo of a tree", size_hint=(.96, .94), **kwargs)
        self._callback = callback
        self._done = False
        root = BoxLayout(orientation="vertical", spacing=dp(6))

        # Shortcut buttons to the usual photo folders
        shortcuts = BoxLayout(size_hint_x=None, spacing=dp(6))
        shortcuts.bind(minimum_width=shortcuts.setter("width"))
        home = os.path.expanduser("~")
        places = [("Home", home)]
        for name in ("Pictures", "DCIM", "Downloads", "Desktop", "Documents"):
            path = os.path.join(home, name)
            if os.path.isdir(path):
                places.append((name, path))
        if platform == "android":
            for name, path in (("Camera", "/storage/emulated/0/DCIM/Camera"),
                               ("Pictures", "/storage/emulated/0/Pictures"),
                               ("Downloads", "/storage/emulated/0/Download")):
                if os.path.isdir(path):
                    places.append((name, path))
        for name, path in places:
            b = Button(text=name, size_hint_x=None, width=dp(100))
            b.bind(on_release=lambda _b, p=path: self._go(p))
            shortcuts.add_widget(b)
        scroll = ScrollView(size_hint_y=None, height=dp(44), do_scroll_y=False)
        scroll.add_widget(shortcuts)
        root.add_widget(scroll)

        # Current folder + Up
        nav = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        up = Button(text="Up", size_hint_x=None, width=dp(70))
        self.path_label = Label(text="", halign="left", valign="middle", shorten=True,
                                shorten_from="left")
        self.path_label.bind(size=lambda l, s: setattr(l, "text_size", s))
        nav.add_widget(up)
        nav.add_widget(self.path_label)
        root.add_widget(nav)

        # Files + preview
        body = BoxLayout(orientation="vertical" if platform == "android" else "horizontal",
                         spacing=dp(6))
        self.chooser = FileChooserIconView(path=start_folder(), filters=_patterns)
        self.preview = Image(fit_mode="contain", size_hint=(.45, 1) if platform != "android" else (1, .35))
        body.add_widget(self.chooser)
        body.add_widget(self.preview)
        root.add_widget(body)

        # Buttons
        buttons = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(6))
        cancel = Button(text="Cancel")
        self.open_btn = Button(text="Open", disabled=True, bold=True)
        buttons.add_widget(cancel)
        buttons.add_widget(self.open_btn)
        root.add_widget(buttons)
        self.content = root

        up.bind(on_release=lambda *_: self._go(os.path.dirname(self.chooser.path.rstrip(os.sep)) or os.sep))
        self.chooser.bind(path=lambda *_: self._update_path(), selection=lambda *_: self._selected())
        self.chooser.bind(on_submit=lambda *_: self._open())
        cancel.bind(on_release=lambda *_: self.dismiss())
        self.open_btn.bind(on_release=lambda *_: self._open())
        self.bind(on_dismiss=lambda *_: None if self._done else _later(self._callback, None))
        self._update_path()

    def _go(self, path):
        if os.path.isdir(path):
            self.chooser.path = path

    def _update_path(self):
        self.path_label.text = self.chooser.path

    def _selected(self):
        sel = self.chooser.selection
        ok = bool(sel) and sel[0].lower().endswith(IMAGE_EXTS)
        self.open_btn.disabled = not ok
        self.preview.source = sel[0] if ok else ""

    def _open(self):
        sel = self.chooser.selection
        if sel and sel[0].lower().endswith(IMAGE_EXTS):
            self._done = True
            self.dismiss()
            _later(self._callback, sel[0])
