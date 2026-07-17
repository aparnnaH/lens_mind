from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from lensmind.db.models import Photo


@dataclass(frozen=True)
class PhotoDisplayInfo:
    preview_path: Path | None
    filename: str
    original_path: Path
    file_size: int
    capture_date: datetime | None
    timestamp_source: str | None
    dimensions: tuple[int, int] | None
    camera_details: str | None
    gps_coordinates: tuple[float, float] | None
    blur_score: float | None
    source_folder: Path | None
    missing_file: bool


class PhotoDisplayService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_photo_display_info(self, photo_id: int) -> PhotoDisplayInfo | None:
        photo = self._session.get(Photo, photo_id)
        if photo is None:
            return None

        return PhotoDisplayInfo(
            preview_path=_preview_path(photo),
            filename=photo.filename,
            original_path=Path(photo.original_path),
            file_size=photo.file_size,
            capture_date=photo.capture_timestamp,
            timestamp_source=photo.timestamp_source,
            dimensions=_dimensions(photo),
            camera_details=_camera_details(photo),
            gps_coordinates=_gps_coordinates(photo),
            blur_score=photo.blur_score,
            source_folder=_source_folder_path(photo),
            missing_file=photo.missing_file,
        )


def _preview_path(photo: Photo) -> Path | None:
    if photo.thumbnail_path:
        return Path(photo.thumbnail_path)
    if photo.missing_file:
        return None
    return Path(photo.original_path)


def _dimensions(photo: Photo) -> tuple[int, int] | None:
    if photo.width is None or photo.height is None:
        return None
    return (photo.width, photo.height)


def _camera_details(photo: Photo) -> str | None:
    parts = [part for part in (photo.camera_make, photo.camera_model) if part]
    if not parts:
        return None
    return " ".join(parts)


def _gps_coordinates(photo: Photo) -> tuple[float, float] | None:
    if photo.latitude is None or photo.longitude is None:
        return None
    return (photo.latitude, photo.longitude)


def _source_folder_path(photo: Photo) -> Path | None:
    if photo.source_folder is None:
        return None
    return Path(photo.source_folder.path)
