import os

from tree_divider import exporter, storage


def test_saves_into_pictures_treedivider(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "pictures_dir", lambda: str(tmp_path / "Pictures"))
    where, folder, normal = storage.write_picture(b"jpegdata", "a.jpg")
    assert normal is True
    assert folder == str(tmp_path / "Pictures" / "TreeDivider")
    assert where == os.path.join(folder, "a.jpg")
    assert open(where, "rb").read() == b"jpegdata"


def test_falls_back_and_reports_it_when_pictures_unusable(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")                       # a file where the folder should be
    monkeypatch.setattr(storage, "pictures_dir", lambda: str(blocker))
    where, folder, normal = storage.write_picture(b"x", "b.jpg", fallback_root=str(tmp_path / "app"))
    assert normal is False
    assert folder == str(tmp_path / "app" / "TreeDivider")
    assert os.path.exists(where)


def test_no_test_file_left_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "pictures_dir", lambda: str(tmp_path))
    storage.write_picture(b"x", "c.jpg")
    assert os.listdir(tmp_path / "TreeDivider") == ["c.jpg"]


def test_file_name_includes_latest_reading():
    assert exporter.file_name(100, [0.5, 0.1164]).startswith("tree-divider-reading-11_64-")
    assert exporter.file_name(100, []).startswith("tree-divider-2")


def test_fallback_is_not_looked_up_when_pictures_works(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "pictures_dir", lambda: str(tmp_path))
    def broken():
        raise FileNotFoundError("app data folder missing")
    where, _, normal = storage.write_picture(b"x", "d.jpg", fallback_root=broken)
    assert normal and os.path.exists(where)


def test_log_goes_next_to_pictures(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "pictures_dir", lambda: str(tmp_path))
    assert storage.log_folder() == str(tmp_path / "TreeDivider")
