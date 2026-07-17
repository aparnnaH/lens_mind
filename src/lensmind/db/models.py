from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceFolder(Base):
    __tablename__ = "source_folders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    indexing_runs: Mapped[list[IndexingRun]] = relationship(
        back_populates="source_folder",
        cascade="all, delete-orphan",
    )


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    original_path: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source_folder_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_folders.id"),
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(16), index=True)
    capture_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timestamp_source: Mapped[str | None] = mapped_column(String)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    camera_make: Mapped[str | None] = mapped_column(String)
    camera_model: Mapped[str | None] = mapped_column(String)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    blur_score: Mapped[float | None] = mapped_column(Float)
    thumbnail_path: Mapped[str | None] = mapped_column(String)
    processing_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
    )
    processing_error: Mapped[str | None] = mapped_column(Text)
    missing_file: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    album_links: Mapped[list[AlbumPhoto]] = relationship(
        back_populates="photo",
        cascade="all, delete-orphan",
    )
    trip_links: Mapped[list[TripPhoto]] = relationship(
        back_populates="photo",
        cascade="all, delete-orphan",
    )
    duplicate_links: Mapped[list[DuplicateGroupPhoto]] = relationship(
        back_populates="photo",
        cascade="all, delete-orphan",
    )
    embeddings: Mapped[list[PhotoEmbedding]] = relationship(
        back_populates="photo",
        cascade="all, delete-orphan",
    )
    source_folder: Mapped[SourceFolder | None] = relationship()


class Album(Base):
    __tablename__ = "albums"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    cover_photo_id: Mapped[int | None] = mapped_column(ForeignKey("photos.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    photo_links: Mapped[list[AlbumPhoto]] = relationship(
        back_populates="album",
        cascade="all, delete-orphan",
    )
    cover_photo: Mapped[Photo | None] = relationship(foreign_keys=[cover_photo_id])


class AlbumPhoto(Base):
    __tablename__ = "album_photos"

    album_id: Mapped[int] = mapped_column(ForeignKey("albums.id"), primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    album: Mapped[Album] = relationship(back_populates="photo_links")
    photo: Mapped[Photo] = relationship(back_populates="album_links")


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    cover_photo_id: Mapped[int | None] = mapped_column(ForeignKey("photos.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    photo_links: Mapped[list[TripPhoto]] = relationship(
        back_populates="trip",
        cascade="all, delete-orphan",
    )
    cover_photo: Mapped[Photo | None] = relationship(foreign_keys=[cover_photo_id])


class TripPhoto(Base):
    __tablename__ = "trip_photos"

    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    trip: Mapped[Trip] = relationship(back_populates="photo_links")
    photo: Mapped[Photo] = relationship(back_populates="trip_links")


class IndexingRun(Base):
    __tablename__ = "indexing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_folder_id: Mapped[int] = mapped_column(
        ForeignKey("source_folders.id"),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    files_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    source_folder: Mapped[SourceFolder] = relationship(back_populates="indexing_runs")


class DuplicateGroup(Base):
    __tablename__ = "duplicate_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    classification: Mapped[str] = mapped_column(String, nullable=False)
    group_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    distance_threshold: Mapped[int | None] = mapped_column(Integer)
    max_distance: Mapped[int | None] = mapped_column(Integer)
    reviewed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_photo_id: Mapped[int | None] = mapped_column(ForeignKey("photos.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    photo_links: Mapped[list[DuplicateGroupPhoto]] = relationship(
        back_populates="duplicate_group",
        cascade="all, delete-orphan",
    )
    preferred_photo: Mapped[Photo | None] = relationship(
        foreign_keys=[preferred_photo_id],
    )


class DuplicateGroupPhoto(Base):
    __tablename__ = "duplicate_group_photos"

    duplicate_group_id: Mapped[int] = mapped_column(
        ForeignKey("duplicate_groups.id"),
        primary_key=True,
    )
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), primary_key=True)

    duplicate_group: Mapped[DuplicateGroup] = relationship(
        back_populates="photo_links",
    )
    photo: Mapped[Photo] = relationship(back_populates="duplicate_links")


class PhotoEmbedding(Base):
    __tablename__ = "photo_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), nullable=False)
    photo_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_config: Mapped[str] = mapped_column(String, nullable=False)
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_reference: Mapped[str | None] = mapped_column(String)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    photo: Mapped[Photo] = relationship(back_populates="embeddings")
