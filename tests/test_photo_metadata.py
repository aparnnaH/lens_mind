from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from lensmind.services.photo_metadata import PhotoMetadataExtractor

EXIF_TAGS = {value: key for key, value in ExifTags.TAGS.items()}
GPS_TAGS = {value: key for key, value in ExifTags.GPSTAGS.items()}


def test_extracts_exif_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (640, 480), color="white")
    exif = Image.Exif()
    exif[EXIF_TAGS["DateTimeOriginal"]] = "2026:02:03 04:05:06"
    exif[EXIF_TAGS["Orientation"]] = 6
    exif[EXIF_TAGS["Make"]] = "Canon"
    exif[EXIF_TAGS["Model"]] = "EOS"
    gps_ifd = {
        GPS_TAGS["GPSLatitudeRef"]: "N",
        GPS_TAGS["GPSLatitude"]: (43.0, 38.0, 0.0),
        GPS_TAGS["GPSLongitudeRef"]: "W",
        GPS_TAGS["GPSLongitude"]: (79.0, 23.0, 0.0),
    }
    exif[EXIF_TAGS["GPSInfo"]] = gps_ifd
    image.save(image_path, exif=exif)

    metadata = PhotoMetadataExtractor().extract(image_path)

    assert metadata.error is None
    assert metadata.capture_timestamp == datetime(2026, 2, 3, 4, 5, 6)
    assert metadata.timestamp_source == "exif"
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.orientation == 6
    assert metadata.camera_make == "Canon"
    assert metadata.camera_model == "EOS"
    assert metadata.latitude == 43 + (38 / 60)
    assert metadata.longitude == -(79 + (23 / 60))


def test_uses_filesystem_timestamp_when_exif_date_is_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (100, 50), color="blue").save(image_path)
    expected_timestamp = datetime(2026, 1, 2, 3, 4, 5).timestamp()
    os.utime(image_path, (expected_timestamp, expected_timestamp))

    metadata = PhotoMetadataExtractor().extract(image_path)

    assert metadata.error is None
    assert metadata.capture_timestamp == datetime.fromtimestamp(expected_timestamp)
    assert metadata.timestamp_source == "filesystem"
    assert metadata.width == 100
    assert metadata.height == 50


def test_corrupted_images_return_error_records(tmp_path: Path) -> None:
    bad_image = tmp_path / "bad.jpg"
    good_image = tmp_path / "good.jpg"
    bad_image.write_bytes(b"not an image")
    Image.new("RGB", (10, 20), color="red").save(good_image)

    records = PhotoMetadataExtractor().extract_many([bad_image, good_image])

    assert records[0].path == bad_image
    assert records[0].error is not None
    assert records[1].path == good_image
    assert records[1].error is None
    assert records[1].width == 10
    assert records[1].height == 20
