from pathlib import Path

from ejo_vision.io import load_image_paths


def test_missing_directory_returns_empty_list(tmp_path: Path):
    assert load_image_paths(tmp_path / "does_not_exist") == []


def test_sorts_zero_padded_filenames(tmp_path: Path):
    (tmp_path / "000002.png").write_bytes(b"")
    (tmp_path / "000010.png").write_bytes(b"")
    (tmp_path / "000001.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore")
    names = [p.name for p in load_image_paths(tmp_path)]
    assert names == ["000001.png", "000002.png", "000010.png"]
