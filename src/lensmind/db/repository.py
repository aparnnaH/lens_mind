from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.models import Base, IndexingRun, Photo, SourceFolder


@dataclass(frozen=True)
class PhotoData:
    original_path: str
    filename: str
    file_size: int
    sha256: str | None = None
    capture_timestamp: datetime | None = None
    width: int | None = None
    height: int | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    blur_score: float | None = None
    thumbnail_path: str | None = None
    processing_status: str = "pending"
    processing_error: str | None = None
    missing_file: bool = False


def initialize_sqlite(database_path: Path | str) -> sessionmaker[Session]:
    engine = create_sqlite_engine(database_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_sqlite_engine(database_path: Path | str) -> Engine:
    path = Path(database_path).expanduser()
    return create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
        future=True,
    )


class PhotoRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_source_folder(self, path: str) -> SourceFolder:
        existing = self._session.scalar(
            select(SourceFolder).where(SourceFolder.path == path),
        )
        if existing is not None:
            return existing

        source_folder = SourceFolder(path=path)
        self._session.add(source_folder)
        self._session.commit()
        return source_folder

    def add_or_update_photo(self, data: PhotoData) -> Photo:
        photo = self._session.scalar(
            select(Photo).where(Photo.original_path == data.original_path),
        )
        if photo is None:
            photo = Photo(
                original_path=data.original_path,
                filename=data.filename,
                file_size=data.file_size,
            )
            self._session.add(photo)

        photo.filename = data.filename
        photo.file_size = data.file_size
        photo.sha256 = data.sha256
        photo.capture_timestamp = data.capture_timestamp
        photo.width = data.width
        photo.height = data.height
        photo.camera_make = data.camera_make
        photo.camera_model = data.camera_model
        photo.latitude = data.latitude
        photo.longitude = data.longitude
        photo.blur_score = data.blur_score
        photo.thumbnail_path = data.thumbnail_path
        photo.processing_status = data.processing_status
        photo.processing_error = data.processing_error
        photo.missing_file = data.missing_file

        self._session.commit()
        return photo

    def list_photos(self) -> list[Photo]:
        return list(self._session.scalars(select(Photo).order_by(Photo.id)))

    def mark_photo_missing(self, photo_id: int) -> Photo | None:
        photo = self._session.get(Photo, photo_id)
        if photo is None:
            return None

        photo.missing_file = True
        self._session.commit()
        return photo

    def record_indexing_run(
        self,
        source_folder_id: int,
        *,
        status: str,
        files_seen: int = 0,
        files_added: int = 0,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> IndexingRun:
        indexing_run = IndexingRun(
            source_folder_id=source_folder_id,
            status=status,
            files_seen=files_seen,
            files_added=files_added,
            finished_at=finished_at,
            error=error,
        )
        if started_at is not None:
            indexing_run.started_at = started_at

        self._session.add(indexing_run)
        self._session.commit()
        return indexing_run
