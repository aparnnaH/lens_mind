from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from PIL import ExifTags, Image, UnidentifiedImageError

TimestampSource = Literal["exif", "filesystem"]

EXIF_TAGS = {value: key for key, value in ExifTags.TAGS.items()}
GPS_TAGS = {value: key for key, value in ExifTags.GPSTAGS.items()}


@dataclass(frozen=True)
class PhotoMetadata:
    path: Path
    capture_timestamp: datetime | None
    timestamp_source: TimestampSource | None
    width: int | None
    height: int | None
    orientation: int | None
    camera_make: str | None
    camera_model: str | None
    latitude: float | None
    longitude: float | None
    error: str | None = None


class PhotoMetadataExtractor:
    def extract(self, path: Path | str) -> PhotoMetadata:
        image_path = Path(path)

        try:
            with Image.open(image_path) as image:
                exif = image.getexif()
                capture_timestamp = _extract_capture_timestamp(exif)
                timestamp_source: TimestampSource | None = "exif"
                if capture_timestamp is None:
                    capture_timestamp = _filesystem_timestamp(image_path)
                    timestamp_source = "filesystem" if capture_timestamp else None

                return PhotoMetadata(
                    path=image_path,
                    capture_timestamp=capture_timestamp,
                    timestamp_source=timestamp_source,
                    width=image.width,
                    height=image.height,
                    orientation=_optional_int(exif.get(EXIF_TAGS["Orientation"])),
                    camera_make=_optional_str(exif.get(EXIF_TAGS["Make"])),
                    camera_model=_optional_str(exif.get(EXIF_TAGS["Model"])),
                    latitude=_extract_gps_coordinate(exif, "GPSLatitude"),
                    longitude=_extract_gps_coordinate(exif, "GPSLongitude"),
                )
        except (OSError, UnidentifiedImageError) as error:
            return PhotoMetadata(
                path=image_path,
                capture_timestamp=None,
                timestamp_source=None,
                width=None,
                height=None,
                orientation=None,
                camera_make=None,
                camera_model=None,
                latitude=None,
                longitude=None,
                error=str(error),
            )

    def extract_many(self, paths: Iterable[Path | str]) -> list[PhotoMetadata]:
        return [self.extract(path) for path in paths]


def _extract_capture_timestamp(exif: Image.Exif) -> datetime | None:
    date_value = exif.get(EXIF_TAGS["DateTimeOriginal"]) or exif.get(
        EXIF_TAGS["DateTime"],
    )
    if not isinstance(date_value, str):
        return None

    try:
        return datetime.strptime(date_value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _filesystem_timestamp(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _extract_gps_coordinate(exif: Image.Exif, coordinate_tag: str) -> float | None:
    gps_info = exif.get_ifd(EXIF_TAGS["GPSInfo"])
    coordinate = gps_info.get(GPS_TAGS[coordinate_tag])
    reference = gps_info.get(GPS_TAGS[f"{coordinate_tag}Ref"])
    if coordinate is None or reference is None:
        return None

    decimal_coordinate = _degrees_to_decimal(coordinate)
    if decimal_coordinate is None:
        return None

    if reference in {"S", "W"}:
        return -decimal_coordinate
    return decimal_coordinate


def _degrees_to_decimal(values: object) -> float | None:
    if not isinstance(values, tuple) or len(values) != 3:
        return None

    degrees = _ratio_to_float(values[0])
    minutes = _ratio_to_float(values[1])
    seconds = _ratio_to_float(values[2])
    if degrees is None or minutes is None or seconds is None:
        return None

    return degrees + (minutes / 60) + (seconds / 3600)


def _ratio_to_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
        return float(value[0]) / float(value[1])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
