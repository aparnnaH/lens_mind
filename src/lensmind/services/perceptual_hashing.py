from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class PerceptualHashResult:
    path: Path
    value: str | None
    error: str | None = None


class PerceptualHashService:
    def __init__(self, hash_size: int = 8) -> None:
        self._hash_size = hash_size

    def compute(self, path: Path | str) -> PerceptualHashResult:
        image_path = Path(path)
        try:
            with Image.open(image_path) as image:
                normalized = ImageOps.exif_transpose(image).convert("L")
                normalized = normalized.resize(
                    (self._hash_size + 1, self._hash_size),
                    Image.Resampling.LANCZOS,
                )
                get_pixels = getattr(
                    normalized,
                    "get_flattened_data",
                    normalized.getdata,
                )
                pixels = list(get_pixels())
        except (OSError, UnidentifiedImageError) as error:
            return PerceptualHashResult(
                path=image_path,
                value=None,
                error=str(error),
            )

        bits = []
        row_width = self._hash_size + 1
        for y in range(self._hash_size):
            for x in range(self._hash_size):
                left = pixels[(y * row_width) + x]
                right = pixels[(y * row_width) + x + 1]
                bits.append("1" if left > right else "0")

        return PerceptualHashResult(
            path=image_path,
            value=f"{int(''.join(bits), 2):0{self._hash_size * self._hash_size // 4}x}",
        )


def hamming_distance(first_hash: str, second_hash: str) -> int:
    first_value = int(first_hash, 16)
    second_value = int(second_hash, 16)
    return (first_value ^ second_value).bit_count()
