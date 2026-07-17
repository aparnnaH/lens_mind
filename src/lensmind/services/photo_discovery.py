from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_PHOTO_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".heic",
    },
)


@dataclass(frozen=True)
class PhotoFileRecord:
    path: Path
    filename: str
    file_size: int
    extension: str


def discover_photo_files(
    folder: Path | str,
    supported_extensions: Collection[str] = SUPPORTED_PHOTO_EXTENSIONS,
) -> list[PhotoFileRecord]:
    root = Path(folder).expanduser()
    extensions = {extension.lower() for extension in supported_extensions}
    records: list[PhotoFileRecord] = []

    for path in _iter_visible_files(root):
        extension = path.suffix.lower()
        if extension not in extensions:
            continue

        try:
            stat_result = path.stat()
        except OSError:
            continue

        records.append(
            PhotoFileRecord(
                path=path,
                filename=path.name,
                file_size=stat_result.st_size,
                extension=extension.removeprefix("."),
            ),
        )

    return records


def _iter_visible_files(folder: Path) -> Iterator[Path]:
    try:
        entries = sorted(folder.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return

    for entry in entries:
        if _is_hidden(entry):
            continue

        try:
            if entry.is_dir():
                yield from _iter_visible_files(entry)
            elif entry.is_file():
                yield entry
        except OSError:
            continue


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)
