"""Main screen: picture with ruler, instruction banner, zoom buttons and bottom controls."""
from __future__ import annotations

import os
import subprocess
import sys
import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.utils import platform

from .. import exporter, height, storage, vision_check
from ..session import Session
from .photo_view import PhotoView  # noqa: F401  (used in KV)
from .picker import pick_image

MAX_TEXTURE = 4096   # display copy only; saving always uses the full-size photo

KV = """
#:set PANEL (0.169, 0.129, 0.094, 0.88)
#:set YELLOW (1, 0.824, 0.247, 1)
#:set INK (0.106, 0.09, 0.071, 1)

<BarButton@Button>:
    background_normal: ""
    background_down: ""
    background_disabled_normal: ""
    background_color: PANEL if self.state == "normal" else YELLOW
    color: (1, 1, 1, 1 if not self.disabled else .35) if self.state == "normal" else INK
    bold: True
    font_size: "14sp"

<TreeDividerScreen>:
    PhotoView:
        id: view

    Label:
        text: root.hint
        bold: True
        font_size: "15sp"
        size_hint: None, None
        text_size: min(root.width - dp(56), dp(520)), None
        halign: "center"
        size: min(self.texture_size[0], root.width - dp(56)) + dp(32), max(dp(40), self.texture_size[1] + dp(16))
        pos_hint: {"center_x": .5, "top": .98}
        canvas.before:
            Color:
                rgba: PANEL
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [dp(20)]

    # Tree height (tap to change the banner height)
    Button:
        text: root.height_text
        markup: True
        halign: "left"
        background_normal: ""
        background_down: ""
        background_color: PANEL
        size_hint: None, None
        size: self.texture_size[0] + dp(24), self.texture_size[1] + dp(14)
        pos_hint: {"x": .02, "top": .9}
        opacity: 1 if root.height_text else 0
        disabled: not root.height_text
        on_release: root.edit_banner()

    # Zoom controls
    BoxLayout:
        orientation: "vertical"
        size_hint: None, None
        size: dp(52), dp(212)
        spacing: dp(4)
        pos_hint: {"right": .98, "center_y": .55}
        opacity: 1 if root.has_picture else 0
        disabled: not root.has_picture
        BarButton:
            text: "+"
            font_size: "24sp"
            on_release: view.zoom_by(1.5)
        Label:
            text: "{}%".format(int(view.zoom_percent))
            bold: True
            font_size: "12sp"
            size_hint_y: None
            height: dp(28)
            canvas.before:
                Color:
                    rgba: PANEL
                Rectangle:
                    pos: self.pos
                    size: self.size
        BarButton:
            text: "-"
            font_size: "24sp"
            on_release: view.zoom_by(1 / 1.5)
        BarButton:
            text: "Fit"
            on_release: view.fit()

    BoxLayout:
        size_hint: 1, None
        height: dp(56)
        padding: dp(8)
        spacing: dp(6)
        BarButton:
            text: "Camera" if root.using_photo else ("Live" if root.frozen else "Freeze")
            disabled: not root.has_camera
            on_release: root.camera_button()
        BarButton:
            text: "Photo"
            on_release: root.open_photo()
        BarButton:
            text: "Auto"
            on_release: root.auto_dialog()
        BarButton:
            text: "-"
            size_hint_x: .5
            on_release: root.step_parts(-1)
        Label:
            text: str(root.parts)
            bold: True
            font_size: "18sp"
            size_hint_x: .6
        BarButton:
            text: "+"
            size_hint_x: .5
            on_release: root.step_parts(1)
        BarButton:
            text: "Save"
            on_release: root.save()
        BarButton:
            text: "Reset"
            on_release: root.reset()
"""


def texture_from_pil(img):
    """Kivy texture for showing a Pillow image (downsized if the GPU can't take it)."""
    disp = img.convert("RGBA")
    if max(disp.size) > MAX_TEXTURE:
        disp.thumbnail((MAX_TEXTURE, MAX_TEXTURE))
    tex = Texture.create(size=disp.size, colorfmt="rgba")
    tex.blit_buffer(disp.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
    tex.flip_vertical()             # Pillow rows start at the top, OpenGL at the bottom
    return tex


class TreeDividerScreen(FloatLayout):
    hint = StringProperty("Starting camera...")
    parts = NumericProperty(100)
    frozen = BooleanProperty(False)
    using_photo = BooleanProperty(False)
    has_camera = BooleanProperty(False)
    has_picture = BooleanProperty(False)
    height_text = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = Session(parts=100, hit_radius=dp(28), read_radius=dp(70))
        self.camera = None
        self.picture = None          # full-size Pillow image when a photo is open
        self.last_saved = None
        self.last_log = None
        # Machine vision (optional: needs OpenCV; OCR also needs Tesseract)
        self.vision_ok = vision_check.available()
        self.ocr_ok = False
        if self.vision_ok:
            from .. import vision
            self.ocr_ok = vision.ocr_available()
        self.vision_gen = 0          # bumps on every change, so stale results are ignored
        self.photo_gen = 0
        self._vision_event = None
        self._auto = None             # automatic measurer, loaded on first use
        self._auto_busy = False
        stored = self._settings_get("banner_m")
        if isinstance(stored, (int, float)) and stored > 0:
            self.session.banner_m = float(stored)
        view = self.ids.view
        view.session = self.session
        view.on_event = self.on_session_event

    @property
    def view(self):
        return self.ids.view

    # ---- camera -------------------------------------------------------------------
    def start_camera(self, *_):
        from kivy.uix.camera import Camera
        try:
            cam = Camera(index=0, resolution=(1280, 720), play=True,
                         size_hint=(None, None), size=(1, 1), opacity=0)
        except Exception as exc:
            cam = None
            Logger.warning(f"TreeDivider: camera unavailable: {exc}")
        if cam is None or getattr(cam, "_camera", None) is None:
            if not self.using_photo:
                self.hint = "No camera found - tap Photo to open a picture"
            return
        self.camera = cam
        self.add_widget(cam, index=len(self.children))   # hidden; it only feeds frames
        self.has_camera = True
        cam.bind(texture=self._on_camera_texture)
        Clock.schedule_interval(self._tick, 1 / 30)
        if cam.texture is not None:
            self._on_camera_texture(cam, cam.texture)

    def _on_camera_texture(self, cam, tex):
        if tex is not None and not self.using_photo:
            self.view.set_picture(tex, tex.width, tex.height)
            self.has_picture = True
            self.refresh()

    def _tick(self, _dt):
        if self.camera and self.camera.play and not self.using_photo:
            self.view.canvas.ask_update()

    def camera_button(self):
        if not self.camera:
            return
        if self.using_photo:                      # back from a photo to the live camera
            self.using_photo = False
            self.picture = None
            self.session.reset()
            self.frozen = False
            self.camera.play = True
            if self.camera.texture is not None:
                self._on_camera_texture(self.camera, self.camera.texture)
            self.view.fit()
            self.schedule_vision()
        else:
            self.frozen = not self.frozen
            self.camera.play = not self.frozen
        self.refresh()

    # ---- photo -----------------------------------------------------------------------
    def open_photo(self):
        pick_image(self._photo_chosen)

    def _photo_chosen(self, path):
        if path:
            self.load_photo(path)

    def load_photo(self, path):
        from PIL import Image as PILImage, ImageOps
        try:
            img = ImageOps.exif_transpose(PILImage.open(path))   # phone photos: fix rotation
            img.load()
        except Exception as exc:
            Logger.warning(f"TreeDivider: could not open {path}: {exc}")
            self.hint = "That file couldn't be opened as a picture"
            return
        self.picture = img
        if self.camera:
            self.camera.play = False
        self.using_photo = True
        self.frozen = False
        self.session.reset()
        self.view.viewport = None                 # new photo: start fitted
        self.view.set_picture(texture_from_pil(img), img.width, img.height)
        self.has_picture = True
        self.refresh()
        self.schedule_vision()
        self._look_for_ruler(img)

    # ---- the current picture -------------------------------------------------------------
    def _current_picture(self):
        """Full-size Pillow image of what's on screen (call on the main thread)."""
        if self.using_photo and self.picture is not None:
            return self.picture
        if self.camera and self.camera.texture:
            tex = self.camera.texture
            return exporter.image_from_rgba(tex.pixels, tex.width, tex.height)
        return None

    # ---- machine vision: check every reading -------------------------------------------
    def schedule_vision(self, *_):
        """Re-read the banners with machine vision shortly after any change."""
        self.vision_gen += 1
        s, view = self.session, self.view
        if self._vision_event:
            self._vision_event.cancel()
            self._vision_event = None
        if not self.vision_ok or s.ruler is None or not s.readings:
            view.vision_labels = []
            view.redraw()
            return
        view.vision_labels = [vision_check.PENDING] * len(s.readings)
        view.redraw()
        gen = self.vision_gen
        self._vision_event = Clock.schedule_once(lambda *_: self._run_vision(gen), 0.35)

    def _run_vision(self, gen):
        if gen != self.vision_gen:
            return
        picture = self._current_picture()
        s = self.session
        if picture is None or s.ruler is None:
            return
        args = (picture, s.base, s.top, s.parts, list(s.readings))
        use_ocr = self.ocr_ok

        def work():
            try:
                checks = vision_check.check_readings(*args, use_ocr=use_ocr)
            except Exception:
                Logger.exception("TreeDivider: vision check failed")
                checks = [vision_check.Check("Vision: error", "not_found")] * len(args[4])
            Clock.schedule_once(lambda *_: self._vision_done(gen, checks))

        threading.Thread(target=work, daemon=True).start()

    def _vision_done(self, gen, checks):
        if gen == self.vision_gen:           # ignore results for an older state
            self.view.vision_labels = checks
            self.view.redraw()

    # ---- machine vision: rulers already in an opened picture ----------------------------
    def _look_for_ruler(self, picture):
        if not self.vision_ok:
            return
        self.photo_gen += 1
        gen, parts = self.photo_gen, self.session.parts

        def work():
            try:
                found = vision_check.find_existing_ruler(picture, parts)
            except Exception:
                Logger.exception("TreeDivider: ruler search failed")
                found = None
            if found:
                Clock.schedule_once(lambda *_: self._offer_found(gen, found))

        threading.Thread(target=work, daemon=True).start()

    def _offer_found(self, gen, found):
        if gen != self.photo_gen or not self.using_photo or self.session.base is not None:
            return                            # another photo opened, or user already tapping
        n = len(found.values)
        values = ", ".join(f"{v:.2f}" for v in found.values)
        parts = (f"It was saved with {found.parts} parts (worked out from the banner)."
                 if found.parts_guessed else f"Assuming {found.parts} parts, the app's current setting.")
        message("Ruler found in this picture",
                f"This picture already has a Tree Divider ruler with reading{'s' if n > 1 else ''} "
                f"{values}.\n{parts}\n\nLoad it so you can check or adjust it?",
                action=lambda: self._load_found(found), action_label="Load ruler")

    def _load_found(self, found):
        s = self.session
        s.reset()
        s.set_parts(found.parts)
        s.base, s.top = found.base, found.top
        s.readings = list(found.fractions)
        self.refresh()
        self.schedule_vision()

    # ---- controls ---------------------------------------------------------------------
    def step_parts(self, direction):
        self.session.step_parts(direction)
        self.refresh()
        self.schedule_vision()

    def reset(self):
        self.session.reset()
        if self.camera and not self.using_photo:
            self.frozen = False
            self.camera.play = True
        self.refresh()
        self.schedule_vision()

    def on_session_event(self, event):
        # Freeze the camera as soon as the base is tapped so the ruler stays on the tree.
        if event == "base_set" and self.camera and not self.using_photo and not self.frozen:
            self.frozen = True
            self.camera.play = False
        self.refresh()
        if event in ("top_set", "reading_added", "reading_removed", "moved"):
            self.schedule_vision()

    def refresh(self):
        s = self.session
        self.parts = s.parts
        if self.has_picture:
            self.hint = s.hint()
        h = s.heights()
        self.view.height_labels = s.height_labels()
        if h:
            self.height_text = (
                f"[size={int(sp(20))}][b][color=FFD23F]Tree height {h.tree_m:.2f} m[/color][/b][/size]\n"
                f"[size={int(sp(12))}]{h.banner_m:g} m banner = {h.banner_reading:.2f} of {h.parts}"
                f"  ·  tap to change[/size]")
        else:
            self.height_text = ""
        self.view.redraw()

    # ---- automatic measurement (Grounding DINO + SAM 2.1) -----------------------------
    @staticmethod
    def _auto_missing():
        """Names of missing packages for automatic measuring (empty if all present)."""
        missing = []
        for mod in ("onnxruntime", "tokenizers", "cv2", "numpy"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        return missing

    def auto_dialog(self):
        if self._auto_busy:
            return
        if self._current_picture() is None:
            message("Automatic measurement", "Open a photo first (Photo button), or start the camera.")
            return
        missing = self._auto_missing()
        if missing:
            message("Automatic measurement: one-time install",
                    "Automatic measuring needs extra packages that this Python doesn't have yet "
                    f"({', '.join(missing)}).\n\nThe app is running with:\n{sys.executable}\n\n"
                    "Click Install now to add them to this Python (needs internet; takes a few minutes).",
                    action=lambda: self._install_auto_packages(missing), action_label="Install now")
            return
        box = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(6))
        info = Label(text="The app finds the tree and the banner by itself and places the dots. "
                          "The banner must stand upright at the base of the tree.",
                     halign="left", valign="top", size_hint_y=None, height=dp(54))
        info.bind(width=lambda l, w: setattr(l, "text_size", (w, None)))
        box.add_widget(info)
        box.add_widget(Label(text="Banner height (metres)", size_hint_y=None, height=dp(22),
                             halign="left", text_size=(dp(300), None)))
        height_in = TextInput(text=f"{self.session.banner_m:g}", input_filter="float",
                              multiline=False, size_hint_y=None, height=dp(42), font_size=sp(17))
        box.add_widget(height_in)
        box.add_widget(Label(text="What the banner looks like (e.g. white board, orange sign)",
                             size_hint_y=None, height=dp(22), halign="left", text_size=(dp(400), None)))
        words_in = TextInput(text=self._settings_get("banner_words") or "banner", multiline=False,
                             size_hint_y=None, height=dp(42), font_size=sp(17))
        box.add_widget(words_in)
        error = Label(text="", color=(1, .55, .2, 1), size_hint_y=None, height=dp(20))
        box.add_widget(error)
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        cancel, go = Button(text="Cancel"), Button(text="Measure", bold=True)
        buttons.add_widget(cancel)
        buttons.add_widget(go)
        box.add_widget(buttons)
        popup = Popup(title="Automatic measurement", content=box, size_hint=(.92, None), height=dp(400))

        def start(*_):
            try:
                banner_m = float(height_in.text)
            except ValueError:
                banner_m = -1
            words = words_in.text.strip().lower()
            if not 0 < banner_m <= 100:
                error.text = "Enter a banner height above 0, e.g. 1"
                return
            if not words:
                error.text = "Describe the banner, e.g. banner or white board"
                return
            self.session.banner_m = banner_m
            self._settings_put("banner_m", banner_m)
            self._settings_put("banner_words", words)
            popup.dismiss()
            self._run_auto(banner_m, words)

        go.bind(on_release=start)
        cancel.bind(on_release=popup.dismiss)
        popup.open()

    def _install_auto_packages(self, missing):
        """pip-install the automatic-measuring packages into the Python running the app."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        req = os.path.join(root, "requirements-auto.txt")
        cmd = [sys.executable, "-m", "pip", "install"] + (
            ["-r", req] if os.path.exists(req) else ["onnxruntime>=1.17", "tokenizers>=0.15"])
        status = Label(text="Installing... this can take a few minutes.", halign="center")
        status.bind(width=lambda l, w: setattr(l, "text_size", (w, None)))
        wait = Popup(title="Installing packages", content=status, size_hint=(.85, None),
                     height=dp(170), auto_dismiss=False)
        wait.open()
        Logger.info(f"TreeDivider: running {' '.join(cmd)}")

        def work():
            try:
                flags = 0x08000000 if sys.platform.startswith("win") else 0   # no console window
                proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
                output, ok = (proc.stdout or "") + (proc.stderr or ""), proc.returncode == 0
            except Exception as exc:
                output, ok = f"{type(exc).__name__}: {exc}", False
            Clock.schedule_once(lambda *_: self._install_done(ok, output, wait))

        threading.Thread(target=work, daemon=True).start()

    def _install_done(self, ok, output, wait):
        import importlib
        wait.dismiss()
        importlib.invalidate_caches()
        still = self._auto_missing()
        Logger.info(f"TreeDivider: pip finished ok={ok}\n{output[-2000:]}")
        if not still:
            message("Installed", "The packages are installed. Click Auto again to measure.")
            return
        tail = "\n".join(l for l in output.strip().splitlines()[-6:])
        hint = ""
        if "no matching distribution" in output.lower() or "could not find a version" in output.lower():
            v = sys.version_info
            if v >= (3, 13):
                hint = (f"\n\nYour Python is version {v.major}.{v.minor}, which may be too new for "
                        "onnxruntime. Installing Python 3.12 from python.org usually fixes this.")
            else:
                hint = ("\n\npip couldn't find the package: check the internet connection "
                        "(a firewall or proxy can block downloads) and try again.")
        elif "externally-managed" in output.lower():
            hint = "\n\nThis Python doesn't allow installs: run the app from its venv instead."
        message("Install didn't finish",
                f"Still missing: {', '.join(still)}\n\nLast messages from pip:\n{tail}{hint}")

    def _run_auto(self, banner_m, words):
        picture = self._current_picture()            # main thread (camera pixels need it)
        if picture is None:
            return
        self._auto_busy = True
        status = Label(text="Loading the models...", halign="center")
        status.bind(width=lambda l, w: setattr(l, "text_size", (w, None)))
        wait = Popup(title="Measuring automatically", content=status, size_hint=(.85, None),
                     height=dp(190), auto_dismiss=False)
        wait.open()
        self.hint = "Measuring automatically..."

        def say(text):
            Clock.schedule_once(lambda *_: setattr(status, "text", text))

        def work():
            res, err = None, None
            try:
                import numpy as np
                from ..auto_measure import AutoMeasurer
                if self._auto is None:
                    say("Loading the models...\n(the first time, about 335 MB is downloaded: "
                        "this can take several minutes)")
                    self._auto = AutoMeasurer(progress=lambda m: say(m))
                say("Finding the tree and the banner...\n(this can take up to a minute)")
                rgb = np.asarray(picture.convert("RGB"))
                res = self._auto.measure(rgb, banner_m, banner_words=(words,))
            except Exception as exc:                       # no internet, broken download...
                Logger.exception("TreeDivider: automatic measurement failed")
                err = exc
            Clock.schedule_once(lambda *_: self._auto_done(res, err, wait))

        threading.Thread(target=work, daemon=True).start()

    def _auto_done(self, res, err, wait):
        self._auto_busy = False
        wait.dismiss()
        if err is not None:
            message("Automatic measurement failed",
                    f"{type(err).__name__}: {err}\n\nThe first run needs an internet connection "
                    "to download the models.")
            self.refresh()
            return
        if res.tree_height_m is None:
            message("Couldn't measure automatically",
                    "\n".join(res.warnings) + "\n\nYou can still click the base, top and banner yourself.")
            self.refresh()
            return
        if res.banner_px >= res.tree_px:
            message("Couldn't measure automatically",
                    f"The banner looks as tall as the tree or taller ({res.banner_px:.0f} px against "
                    f"{res.tree_px:.0f} px), so it can't be at the tree's base: it's probably held "
                    "close to the camera.\n\nStand the banner upright at the base of the tree, the same "
                    "distance from the camera as the trunk, and take the photo again.")
            self.refresh()
            return
        # Put the result in as ordinary dots, so it can be checked and adjusted by hand.
        from ..ruler import Point
        s = self.session
        s.reset()
        s.base = Point(*res.tree_base)
        s.top = Point(*res.tree_top)
        s.readings = [res.banner_px / res.tree_px]      # banner height as a share of the tree
        self.view.fit()
        self.refresh()
        self.schedule_vision()
        text = (f"Tree height: {res.tree_height_m:.2f} m\n"
                f"(tree {res.tree_px:.0f} px, {res.banner_m:g} m banner {res.banner_px:.0f} px)\n\n"
                "The dots have been placed for you. Zoom in to check them and drag any that are "
                "off; the height updates as you move them.")
        if res.warnings:
            text += "\n\nWarnings:\n- " + "\n- ".join(res.warnings)
        message("Automatic measurement" if not res.warnings else "Automatic measurement: please check",
                text)

    # ---- banner height setting ------------------------------------------------------------
    def _settings_store(self):
        """settings.json in the app's data folder (or ~/.treedivider if that can't be made)."""
        from kivy.storage.jsonstore import JsonStore
        try:
            folder = App.get_running_app().user_data_dir
        except Exception:
            folder = os.path.join(os.path.expanduser("~"), ".treedivider")
        try:
            os.makedirs(folder, exist_ok=True)
            return JsonStore(os.path.join(folder, "settings.json"))
        except Exception:
            return None

    def _settings_get(self, key):
        store = self._settings_store()
        try:
            return store.get(key)["value"] if store is not None and store.exists(key) else None
        except Exception:
            return None

    def _settings_put(self, key, value):
        store = self._settings_store()
        try:
            if store is not None:          # (an empty store counts as False)
                store.put(key, value=value)
        except Exception:
            Logger.warning("TreeDivider: couldn't save settings")

    def edit_banner(self):
        box = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(6))
        info = Label(text="Height of the banner (reference object) standing at the base of "
                          "the tree, in metres. The first blue dot marks its top.",
                     halign="left", valign="top", size_hint_y=None, height=dp(60))
        info.bind(width=lambda l, w: setattr(l, "text_size", (w, None)))
        field = TextInput(text=f"{self.session.banner_m:g}", input_filter="float", multiline=False,
                          size_hint_y=None, height=dp(44), font_size=sp(18))
        error = Label(text="", color=(1, .55, .2, 1), size_hint_y=None, height=dp(20))
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        cancel, ok = Button(text="Cancel"), Button(text="Save", bold=True)
        buttons.add_widget(cancel)
        buttons.add_widget(ok)
        for w in (info, field, error, buttons):
            box.add_widget(w)
        popup = Popup(title="Banner height", content=box, size_hint=(.9, None), height=dp(290))

        def save(*_):
            try:
                value = float(field.text)
            except ValueError:
                value = -1
            if not 0 < value <= 100:
                error.text = "Enter a height above 0, e.g. 1 or 1.5"
                return
            self.session.banner_m = value
            self._settings_put("banner_m", value)
            popup.dismiss()
            self.refresh()

        ok.bind(on_release=save)
        field.bind(on_text_validate=save)
        cancel.bind(on_release=popup.dismiss)
        popup.open()

    # ---- saving (Pillow + storage) ------------------------------------------------------
    def save(self):
        s = self.session
        if s.base is None or s.top is None:
            missing = "the base and the top" if s.base is None else "the top"
            message("Nothing saved yet",
                    f"Tap {missing} of the tree first.\nSave keeps the picture with the ruler on it, "
                    "so the ruler has to be there.")
            return
        picture = self._current_picture()
        if picture is None:
            message("Nothing saved yet", "There's no picture to save. Open a photo or start the camera.")
            return
        h = s.heights()
        labels = s.height_labels()
        caption = f"Tree height {h.tree_m:.2f} m  (banner {h.banner_m:g} m)" if h else ""
        name = exporter.file_name(s.parts, s.readings, h.tree_m if h else None)
        try:
            # Points are already in picture pixels, whatever the zoom.
            data = exporter.to_jpeg(picture, s.base, s.top, s.parts, s.readings, labels, caption)
            where, folder, normal = storage.write_picture(
                data, name, fallback_root=lambda: App.get_running_app().user_data_dir)
        except Exception as exc:
            Logger.exception("TreeDivider: save failed")
            message("Couldn't save the picture", f"{type(exc).__name__}: {exc}")
            return
        Logger.info(f"TreeDivider: saved {where}")
        self.last_saved = where
        note = "" if normal else ("\n\nYour Pictures folder couldn't be used, "
                                  "so it went to the app's own folder instead.")
        # Add this picture's readings to readings.txt
        try:
            log_path = os.path.join(
                storage.log_folder(lambda: App.get_running_app().user_data_dir), height.LOG_NAME)
            values = [f * s.parts for f in s.readings]
            height.append_log(log_path, height.log_line(name, h, values, s.parts))
            note += f"\n\nReadings added to:\n{log_path}"
            self.last_log = log_path
        except Exception as exc:
            Logger.exception("TreeDivider: couldn't write readings log")
            note += f"\n\nCouldn't add the readings to {height.LOG_NAME}: {exc}"
        text = f"Picture:\n{where}{note}"
        if h:
            text = f"Tree height: {h.tree_m:.2f} m\n\n{text}"
        open_cb = None
        if folder and storage.can_open_folder():
            open_cb = lambda: storage.open_folder(folder, select_file=where)  # noqa: E731
        message("Picture saved", text, action=open_cb, action_label="Open folder")
        self.hint = f"Saved: {name}"
        Clock.schedule_once(lambda *_: self.refresh(), 6)


def message(title, text, action=None, action_label="OK"):
    """A small window with a message, an OK button and optionally one action button."""
    box = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(6))
    label = Label(text=text, halign="left", valign="top")
    label.bind(width=lambda l, w: setattr(l, "text_size", (w, None)),
               texture_size=lambda l, ts: setattr(l, "height", ts[1]))
    label.size_hint_y = None
    box.add_widget(label)
    buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
    popup = Popup(title=title, content=box, size_hint=(.92, None), auto_dismiss=True)
    if action:
        b = Button(text=action_label, bold=True)
        b.bind(on_release=lambda *_: (popup.dismiss(), action()))
        buttons.add_widget(b)
    ok = Button(text="Not now" if action and action_label != "Open folder" else "OK")
    ok.bind(on_release=popup.dismiss)
    buttons.add_widget(ok)
    box.add_widget(buttons)
    def fit_height(*_):
        popup.height = label.height + dp(48) + dp(10) + dp(110)
    label.bind(height=fit_height)
    fit_height()
    popup.open()
    return popup


class TreeDividerApp(App):
    title = "Tree Divider"

    def build(self):
        Builder.load_string(KV)
        self.screen = TreeDividerScreen()
        return self.screen

    def on_start(self):
        Logger.info(f"TreeDivider: running with Python {sys.version.split()[0]} at {sys.executable}")
        if platform == "android":
            from android.permissions import Permission, request_permissions
            wanted = [Permission.CAMERA]
            for name in ("READ_MEDIA_IMAGES", "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE"):
                if hasattr(Permission, name):
                    wanted.append(getattr(Permission, name))

            def done(permissions, grants):
                Clock.schedule_once(self.screen.start_camera)

            request_permissions(wanted, done)
        else:
            Clock.schedule_once(self.screen.start_camera, 0.2)

    def on_pause(self):
        return True


def run():
    TreeDividerApp().run()
