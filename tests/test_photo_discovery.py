from __future__ import annotations

from pathlib import Path

import pytest

from lensmind.services.photo_discovery import (
    SUPPORTED_PHOTO_EXTENSIONS,
    discover_photo_files,
)


def write_file(path: Path, content: bytes = b"image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_discovers_supported_photos_recursively(tmp_path: Path) -> None:
    write_file(tmp_path / "a.jpg", b"one")
    write_file(tmp_path / "nested" / "b.PNG", b"two")
    write_file(tmp_path / "nested" / "c.webp", b"three")

    records = discover_photo_files(tmp_path)

    assert [record.filename for record in records] == ["a.jpg", "b.PNG", "c.webp"]
    assert [record.file_size for record in records] == [3, 3, 5]
    assert [record.extension for record in records] == ["jpg", "png", "webp"]


def test_supports_expected_photo_extensions_when_available() -> None:
    assert SUPPORTED_PHOTO_EXTENSIONS == {
        ".heic",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    }


def test_ignores_hidden_and_unsupported_files(tmp_path: Path) -> None:
    write_file(tmp_path / "visible.jpeg")
    write_file(tmp_path / "notes.txt")
    write_file(tmp_path / ".hidden.jpg")
    write_file(tmp_path / ".hidden-folder" / "inside.png")

    records = discover_photo_files(tmp_path)

    assert [record.filename for record in records] == ["visible.jpeg"]


def test_continues_after_inaccessible_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_file = tmp_path / "bad.jpg"
    good_file = tmp_path / "good.jpg"
    write_file(bad_file)
    write_file(good_file)
    original_stat = Path.stat

    def fake_stat(path: Path, *, follow_symlinks: bool = True) -> object:
        if path == bad_file:
            raise PermissionError("cannot stat file")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)

    records = discover_photo_files(tmp_path)

    assert [record.filename for record in records] == ["good.jpg"]
