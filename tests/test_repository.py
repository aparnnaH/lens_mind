from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import inspect

from lensmind.db.repository import (
    PhotoData,
    PhotoEmbeddingData,
    PhotoRepository,
    initialize_sqlite,
)


def test_initialize_sqlite_creates_database_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "lensmind.db"
    session_factory = initialize_sqlite(database_path)

    with session_factory() as session:
        table_names = set(inspect(session.bind).get_table_names())

    assert database_path.exists()
    assert {"photos", "source_folders", "indexing_runs"}.issubset(table_names)


def test_add_source_folder_is_idempotent(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        first = repository.add_source_folder("/photos")
        second = repository.add_source_folder("/photos")

    assert first.id == second.id
    assert first.path == "/photos"


def test_add_or_update_photo_and_list_photos(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        created = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/image.jpg",
                filename="image.jpg",
                file_size=1024,
                sha256="a" * 64,
                width=4000,
                height=3000,
            ),
        )
        updated = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/image.jpg",
                filename="renamed.jpg",
                file_size=2048,
                processing_status="indexed",
            ),
        )
        photos = repository.list_photos()

    assert created.id == updated.id
    assert len(photos) == 1
    assert photos[0].filename == "renamed.jpg"
    assert photos[0].file_size == 2048
    assert photos[0].processing_status == "indexed"


def test_mark_photo_missing(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/missing.jpg",
                filename="missing.jpg",
                file_size=512,
            ),
        )
        missing_photo = repository.mark_photo_missing(photo.id)

    assert missing_photo is not None
    assert missing_photo.missing_file is True


def test_list_blurry_photos_filters_by_threshold(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/blurry.jpg",
                filename="blurry.jpg",
                file_size=100,
                blur_score=25.0,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/sharp.jpg",
                filename="sharp.jpg",
                file_size=100,
                blur_score=250.0,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/unknown.jpg",
                filename="unknown.jpg",
                file_size=100,
            ),
        )

        photos = repository.list_blurry_photos(100.0)

    assert [photo.filename for photo in photos] == ["blurry.jpg"]


def test_record_indexing_run(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    finished_at = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)

    with session_factory() as session:
        repository = PhotoRepository(session)
        source_folder = repository.add_source_folder("/photos")
        indexing_run = repository.record_indexing_run(
            source_folder.id,
            status="completed",
            files_seen=10,
            files_added=4,
            started_at=started_at,
            finished_at=finished_at,
        )

    assert indexing_run.id is not None
    assert indexing_run.source_folder_id == source_folder.id
    assert indexing_run.status == "completed"
    assert indexing_run.files_seen == 10
    assert indexing_run.files_added == 4


def test_photo_embedding_cache_uses_photo_hash_and_model_config(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/image.jpg",
                filename="image.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )

        assert repository.photo_needs_embedding(
            photo.id,
            model_name="ViT-B-32",
            model_config="laion2b_s34b_b79k",
        )

        repository.save_photo_embedding(
            PhotoEmbeddingData(
                photo_id=photo.id,
                model_name="ViT-B-32",
                model_config="laion2b_s34b_b79k",
                vector_dimension=3,
                embedding_data=b"embedding-bytes",
            ),
        )
        cached = repository.get_cached_photo_embedding(
            photo.id,
            model_name="ViT-B-32",
            model_config="laion2b_s34b_b79k",
        )

        assert cached is not None
        assert cached.photo_sha256 == "a" * 64
        assert cached.embedding_data == b"embedding-bytes"
        assert cached.embedding_reference is None
        assert cached.vector_dimension == 3
        assert not repository.photo_needs_embedding(
            photo.id,
            model_name="ViT-B-32",
            model_config="laion2b_s34b_b79k",
        )

        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/image.jpg",
                filename="image.jpg",
                file_size=100,
                sha256="b" * 64,
            ),
        )

        assert repository.photo_needs_embedding(
            photo.id,
            model_name="ViT-B-32",
            model_config="laion2b_s34b_b79k",
        )


def test_photo_embedding_can_store_local_reference(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/image.jpg",
                filename="image.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )
        repository.save_photo_embedding(
            PhotoEmbeddingData(
                photo_id=photo.id,
                model_name="ViT-B-32",
                model_config="laion2b_s34b_b79k",
                vector_dimension=512,
                embedding_reference="/embeddings/image.npy",
            ),
        )
        cached = repository.get_cached_photo_embedding(
            photo.id,
            model_name="ViT-B-32",
            model_config="laion2b_s34b_b79k",
        )

    assert cached is not None
    assert cached.embedding_data is None
    assert cached.embedding_reference == "/embeddings/image.npy"
