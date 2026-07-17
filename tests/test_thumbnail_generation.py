from __future__ import annotations

from pathlib import Path

from PIL import ExifTags, Image

from lensmind.services.thumbnail_generation import ThumbnailGenerator

EXIF_TAGS = {value: key for key, value in ExifTags.TAGS.items()}


def create_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_generates_cached_thumbnail_without_modifying_original(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    cache_dir = tmp_path / "cache"
    create_image(source, (800, 600))
    original_bytes = source.read_bytes()

    result = ThumbnailGenerator(cache_dir, size=(128, 128)).generate(source)

    assert result.error is None
    assert result.thumbnail_path is not None
    assert result.thumbnail_path.exists()
    assert result.reused is False
    assert source.read_bytes() == original_bytes

    with Image.open(result.thumbnail_path) as thumbnail:
        assert thumbnail.size == (128, 96)


def test_reuses_cached_thumbnail(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    generator = ThumbnailGenerator(tmp_path / "cache", size=(64, 64))
    create_image(source, (100, 100))

    first = generator.generate(source)
    second = generator.generate(source)

    assert first.error is None
    assert second.error is None
    assert first.thumbnail_path == second.thumbnail_path
    assert first.reused is False
    assert second.reused is True


def test_preserves_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), color="white")
    exif = Image.Exif()
    exif[EXIF_TAGS["Orientation"]] = 6
    image.save(source, exif=exif)

    result = ThumbnailGenerator(tmp_path / "cache", size=(100, 100)).generate(source)

    assert result.thumbnail_path is not None
    with Image.open(result.thumbnail_path) as thumbnail:
        assert thumbnail.size == (20, 40)


def test_handles_corrupted_and_missing_files(tmp_path: Path) -> None:
    corrupted = tmp_path / "bad.jpg"
    missing = tmp_path / "missing.jpg"
    corrupted.write_bytes(b"not an image")
    generator = ThumbnailGenerator(tmp_path / "cache")

    corrupted_result = generator.generate(corrupted)
    missing_result = generator.generate(missing)

    assert corrupted_result.thumbnail_path is None
    assert corrupted_result.error is not None
    assert missing_result.thumbnail_path is None
    assert missing_result.error is not None
