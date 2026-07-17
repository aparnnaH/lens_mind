from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.models import (
    Base,
    DuplicateGroup,
    IndexingRun,
    Photo,
    PhotoEmbedding,
    SourceFolder,
)


@dataclass(frozen=True)
class PhotoData:
    original_path: str
    filename: str
    file_size: int
    source_folder_id: int | None = None
    sha256: str | None = None
    perceptual_hash: str | None = None
    capture_timestamp: datetime | None = None
    timestamp_source: str | None = None
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


@dataclass(frozen=True)
class DuplicatePhotoData:
    id: int
    filename: str
    original_path: str
    file_size: int
    width: int | None
    height: int | None
    blur_score: float | None
    thumbnail_path: str | None
    missing_file: bool


@dataclass(frozen=True)
class DuplicateGroupData:
    id: int
    classification: str
    reviewed: bool
    preferred_photo_id: int | None
    photos: tuple[DuplicatePhotoData, ...]


@dataclass(frozen=True)
class PhotoEmbeddingData:
    photo_id: int
    model_name: str
    model_config: str
    vector_dimension: int
    embedding_data: bytes | None = None
    embedding_reference: str | None = None


@dataclass(frozen=True)
class CachedPhotoEmbeddingData:
    id: int
    photo_id: int
    photo_sha256: str | None
    model_name: str
    model_config: str
    vector_dimension: int
    embedding_data: bytes | None
    embedding_reference: str | None
    generated_at: datetime


@dataclass(frozen=True)
class StoredPhotoEmbeddingData:
    id: int
    photo_id: int
    model_name: str
    model_config: str
    vector_dimension: int
    embedding_data: bytes
    generated_at: datetime


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
        photo.source_folder_id = data.source_folder_id
        photo.file_size = data.file_size
        photo.sha256 = data.sha256
        photo.perceptual_hash = data.perceptual_hash
        photo.capture_timestamp = data.capture_timestamp
        photo.timestamp_source = data.timestamp_source
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

    def list_blurry_photos(self, blur_threshold: float) -> list[Photo]:
        return list(
            self._session.scalars(
                select(Photo)
                .where(Photo.blur_score.is_not(None))
                .where(Photo.blur_score <= blur_threshold)
                .order_by(Photo.id),
            ),
        )

    def list_duplicate_groups(self) -> list[DuplicateGroupData]:
        groups = list(
            self._session.scalars(
                select(DuplicateGroup).order_by(DuplicateGroup.id),
            ),
        )
        return [self._duplicate_group_data(group) for group in groups]

    def save_photo_embedding(self, data: PhotoEmbeddingData) -> PhotoEmbedding:
        photo = self._session.get(Photo, data.photo_id)
        if photo is None:
            msg = f"photo not found: {data.photo_id}"
            raise ValueError(msg)

        existing = self._find_embedding(
            photo_id=data.photo_id,
            photo_sha256=photo.sha256,
            model_name=data.model_name,
            model_config=data.model_config,
        )
        if existing is None:
            existing = PhotoEmbedding(
                photo_id=data.photo_id,
                photo_sha256=photo.sha256,
                model_name=data.model_name,
                model_config=data.model_config,
                vector_dimension=data.vector_dimension,
            )
            self._session.add(existing)

        existing.vector_dimension = data.vector_dimension
        existing.embedding_data = data.embedding_data
        existing.embedding_reference = data.embedding_reference
        self._session.commit()
        return existing

    def get_cached_photo_embedding(
        self,
        photo_id: int,
        *,
        model_name: str,
        model_config: str,
    ) -> CachedPhotoEmbeddingData | None:
        photo = self._session.get(Photo, photo_id)
        if photo is None:
            return None

        embedding = self._find_embedding(
            photo_id=photo_id,
            photo_sha256=photo.sha256,
            model_name=model_name,
            model_config=model_config,
        )
        if embedding is None:
            return None

        return CachedPhotoEmbeddingData(
            id=embedding.id,
            photo_id=embedding.photo_id,
            photo_sha256=embedding.photo_sha256,
            model_name=embedding.model_name,
            model_config=embedding.model_config,
            vector_dimension=embedding.vector_dimension,
            embedding_data=embedding.embedding_data,
            embedding_reference=embedding.embedding_reference,
            generated_at=embedding.generated_at,
        )

    def photo_needs_embedding(
        self,
        photo_id: int,
        *,
        model_name: str,
        model_config: str,
    ) -> bool:
        return (
            self.get_cached_photo_embedding(
                photo_id,
                model_name=model_name,
                model_config=model_config,
            )
            is None
        )

    def list_stored_photo_embeddings(
        self,
        *,
        model_name: str,
        model_config: str,
    ) -> list[StoredPhotoEmbeddingData]:
        embeddings = self._session.scalars(
            select(PhotoEmbedding)
            .where(PhotoEmbedding.model_name == model_name)
            .where(PhotoEmbedding.model_config == model_config)
            .where(PhotoEmbedding.embedding_data.is_not(None))
            .order_by(PhotoEmbedding.photo_id),
        )
        return [
            StoredPhotoEmbeddingData(
                id=embedding.id,
                photo_id=embedding.photo_id,
                model_name=embedding.model_name,
                model_config=embedding.model_config,
                vector_dimension=embedding.vector_dimension,
                embedding_data=embedding.embedding_data or b"",
                generated_at=embedding.generated_at,
            )
            for embedding in embeddings
        ]

    def mark_duplicate_group_reviewed(self, duplicate_group_id: int) -> None:
        duplicate_group = self._session.get(DuplicateGroup, duplicate_group_id)
        if duplicate_group is None:
            return

        duplicate_group.reviewed = True
        self._session.commit()

    def keep_all_duplicate_group_photos(self, duplicate_group_id: int) -> None:
        duplicate_group = self._session.get(DuplicateGroup, duplicate_group_id)
        if duplicate_group is None:
            return

        duplicate_group.preferred_photo_id = None
        duplicate_group.reviewed = True
        self._session.commit()

    def select_preferred_duplicate_photo(
        self,
        duplicate_group_id: int,
        photo_id: int,
    ) -> None:
        duplicate_group = self._session.get(DuplicateGroup, duplicate_group_id)
        if duplicate_group is None:
            return

        linked_photo_ids = {
            photo_link.photo_id for photo_link in duplicate_group.photo_links
        }
        if photo_id not in linked_photo_ids:
            return

        duplicate_group.preferred_photo_id = photo_id
        duplicate_group.reviewed = True
        self._session.commit()

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

    def _duplicate_group_data(
        self,
        duplicate_group: DuplicateGroup,
    ) -> DuplicateGroupData:
        photo_links = sorted(
            duplicate_group.photo_links,
            key=lambda photo_link: photo_link.photo_id,
        )
        photos = tuple(
            DuplicatePhotoData(
                id=photo_link.photo.id,
                filename=photo_link.photo.filename,
                original_path=photo_link.photo.original_path,
                file_size=photo_link.photo.file_size,
                width=photo_link.photo.width,
                height=photo_link.photo.height,
                blur_score=photo_link.photo.blur_score,
                thumbnail_path=photo_link.photo.thumbnail_path,
                missing_file=photo_link.photo.missing_file,
            )
            for photo_link in photo_links
        )
        return DuplicateGroupData(
            id=duplicate_group.id,
            classification=duplicate_group.classification,
            reviewed=duplicate_group.reviewed,
            preferred_photo_id=duplicate_group.preferred_photo_id,
            photos=photos,
        )

    def _find_embedding(
        self,
        *,
        photo_id: int,
        photo_sha256: str | None,
        model_name: str,
        model_config: str,
    ) -> PhotoEmbedding | None:
        return self._session.scalar(
            select(PhotoEmbedding)
            .where(PhotoEmbedding.photo_id == photo_id)
            .where(PhotoEmbedding.photo_sha256 == photo_sha256)
            .where(PhotoEmbedding.model_name == model_name)
            .where(PhotoEmbedding.model_config == model_config),
        )
