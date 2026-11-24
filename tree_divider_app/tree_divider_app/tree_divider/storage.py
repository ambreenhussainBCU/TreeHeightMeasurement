"""
Where saved pictures go, and how to open that place.

  Windows  the real Pictures folder, asked from Windows itself, so OneDrive is
           handled (e.g. C:\\Users\\You\\OneDrive\\Pictures\\TreeDivider)
  Mac      ~/Pictures/TreeDivider
  Linux    your desktop's Pictures folder (xdg-user-dir), else ~/Pictures
  Android  the shared Pictures/TreeDivider folder through MediaStore, so the
           pictures show up in the Gallery / Photos app

If that folder can't be written, a folder in the app's own data is used and
the app says so clearly.
"""
from __future__ import annotations

import os
import subprocess
import sys

APP_FOLDER = "TreeDivider"


def _platform() -> str:
    if "ANDROID_ARGUMENT" in os.environ or "ANDROID_PRIVATE" in os.environ:
        return "android"
    if sys.platform.startswith("win"):
        return "win"
    if sys.platform == "darwin":
        return "macosx"
    return "linux"


# ---- finding the Pictures folder ---------------------------------------------------
def _windows_pictures() -> str | None:
    """Ask Windows for the Pictures folder (follows OneDrive / moved folders)."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    u = uuid.UUID("{33E28130-4E1E-4676-835A-98395C3BC3BB}")   # FOLDERID_Pictures
    guid = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    out = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
        return None
    try:
        return out.value
    finally:
        ctypes.windll.ole32.CoTaskMemFree(out)


def _linux_pictures() -> str | None:
    try:
        path = subprocess.run(["xdg-user-dir", "PICTURES"], capture_output=True,
                              text=True, timeout=3).stdout.strip()
        home = os.path.expanduser("~")
        return path if path and os.path.abspath(path) != os.path.abspath(home) else None
    except (OSError, subprocess.SubprocessError):
        return None


def pictures_dir() -> str:
    plat = _platform()
    found = None
    try:
        if plat == "win":
            found = _windows_pictures()
        elif plat == "linux":
            found = _linux_pictures()
    except Exception:
        found = None
    return found or os.path.join(os.path.expanduser("~"), "Pictures")


def _writable_folder(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "wb") as f:
            f.write(b"ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def save_folder(fallback_root=None) -> tuple[str, bool]:
    """Folder to save into, and whether it's the normal Pictures one (False = fallback).

    `fallback_root` may be a path or a function returning one; it is only used
    (and only looked up) if the Pictures folder can't be written.
    """
    preferred = os.path.join(pictures_dir(), APP_FOLDER)
    if _writable_folder(preferred):
        return preferred, True
    root = None
    try:
        root = fallback_root() if callable(fallback_root) else fallback_root
    except Exception:
        root = None
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".treedivider")
    fallback = os.path.join(root, APP_FOLDER)
    os.makedirs(fallback, exist_ok=True)
    return fallback, False


# ---- writing ---------------------------------------------------------------------------
def _android_save(data: bytes, name: str) -> str:
    """Save into the shared Pictures/TreeDivider folder so the Gallery shows it."""
    from jnius import autoclass

    activity = autoclass("org.kivy.android.PythonActivity").mActivity
    version = autoclass("android.os.Build$VERSION").SDK_INT
    if version >= 29:                       # Android 10+: MediaStore, no permission needed
        ContentValues = autoclass("android.content.ContentValues")
        Columns = autoclass("android.provider.MediaStore$MediaColumns")
        Images = autoclass("android.provider.MediaStore$Images$Media")
        values = ContentValues()
        values.put(Columns.DISPLAY_NAME, name)
        values.put(Columns.MIME_TYPE, "image/jpeg")
        values.put(Columns.RELATIVE_PATH, f"Pictures/{APP_FOLDER}")
        resolver = activity.getContentResolver()
        uri = resolver.insert(Images.EXTERNAL_CONTENT_URI, values)
        if uri is None:
            raise OSError("Android refused to create the picture")
        stream = resolver.openOutputStream(uri)
        try:
            stream.write(bytearray(data))
        finally:
            stream.close()
    else:                                   # Android 9 and older: plain file + media scan
        Environment = autoclass("android.os.Environment")
        base = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath()
        folder = os.path.join(base, APP_FOLDER)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "wb") as f:
            f.write(data)
        Scanner = autoclass("android.media.MediaScannerConnection")
        Scanner.scanFile(activity, [path], None, None)
    return f"Gallery: Pictures/{APP_FOLDER}/{name}"


def write_picture(data: bytes, name: str, fallback_root=None) -> tuple[str, str, bool]:
    """Write JPEG bytes. Returns (where, folder, normal_location)."""
    if _platform() == "android":
        try:
            return _android_save(data, name), "", True
        except Exception:
            pass                            # fall through to app storage below
    folder, normal = save_folder(fallback_root)
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(data)
    return path, folder, normal


# ---- opening the folder -----------------------------------------------------------------
def can_open_folder() -> bool:
    return _platform() in ("win", "macosx", "linux")


def open_folder(folder: str, select_file: str | None = None) -> None:
    plat = _platform()
    if plat == "win":
        if select_file and os.path.exists(select_file):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(select_file)])
        else:
            os.startfile(folder)            # type: ignore[attr-defined]
    elif plat == "macosx":
        subprocess.Popen(["open", "-R", select_file] if select_file else ["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])


# ---- readings log -------------------------------------------------------------------------
def log_folder(fallback_root=None) -> str:
    """Folder for readings.txt: next to the pictures (computer) or Documents/TreeDivider (Android)."""
    if _platform() == "android":
        try:
            from jnius import autoclass
            Environment = autoclass("android.os.Environment")
            docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).getAbsolutePath()
            folder = os.path.join(docs, APP_FOLDER)
            if _writable_folder(folder):
                return folder
        except Exception:
            pass
        root = fallback_root() if callable(fallback_root) else (fallback_root or os.getcwd())
        folder = os.path.join(root, APP_FOLDER)
        os.makedirs(folder, exist_ok=True)
        return folder
    return save_folder(fallback_root)[0]
