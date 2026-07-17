from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from lensmind.services.file_hashing import calculate_sha256


@dataclass(frozen=True)
class ThumbnailResult:
    source_path: Path
    thumbnail_path: Path | None
    reused: bool
    error: str | None = None


class ThumbnailGenerator:
    def __init__(
        self,
        cache_dir: Path | str,
        size: tuple[int, int] = (256, 256),
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._size = size

    def generate(self, source_path: Path | str) -> ThumbnailResult:
        path = Path(source_path)

        try:
            digest = calculate_sha256(path)
        except OSError as error:
            return ThumbnailResult(
                source_path=path,
                thumbnail_path=None,
                reused=False,
                error=str(error),
            )

        thumbnail_path = self._thumbnail_path(digest)
        if thumbnail_path.exists():
            return ThumbnailResult(
                source_path=path,
                thumbnail_path=thumbnail_path,
                reused=True,
            )

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                thumbnail = ImageOps.exif_transpose(image)
                thumbnail.thumbnail(self._size)
                thumbnail.save(thumbnail_path, format="PNG")
        except (OSError, UnidentifiedImageError) as error:
            return ThumbnailResult(
                source_path=path,
                thumbnail_path=None,
                reused=False,
                error=str(error),
            )

        return ThumbnailResult(
            source_path=path,
            thumbnail_path=thumbnail_path,
            reused=False,
        )

    def _thumbnail_path(self, digest: str) -> Path:
        width, height = self._size
        return self._cache_dir / f"{digest}-{width}x{height}.png"
