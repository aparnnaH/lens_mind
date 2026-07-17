from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from lensmind.db.repository import PhotoRepository, initialize_sqlite
from lensmind.services.embeddings import EmbeddingResult
from lensmind.services.file_hashing import calculate_sha256
from lensmind.services.photo_import import PhotoImportService
from lensmind.services.thumbnail_generation import ThumbnailGenerator


def create_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_calculate_sha256(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"lensmind")

    assert calculate_sha256(path) == hashlib.sha256(b"lensmind").hexdigest()


def test_import_folder_saves_discovered_photo_metadata(tmp_path: Path) -> None:
    image_folder = tmp_path / "photos"
    first_image = image_folder / "first.jpg"
    second_image = image_folder / "nested" / "second.png"
    create_image(first_image, (40, 30))
    create_image(second_image, (20, 10))
    (image_folder / "notes.txt").write_text("not a photo")
    create_image(image_folder / ".hidden.jpg", (5, 5))

    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    summary = create_import_service(session_factory, tmp_path).import_folder(
        image_folder,
    )

    with session_factory() as session:
        photos = PhotoRepository(session).list_photos()

    assert summary.files_seen == 2
    assert summary.files_imported == 2
    assert summary.files_added == 2
    assert summary.errors == ()
    assert [photo.filename for photo in photos] == ["first.jpg", "second.png"]
    assert [(photo.width, photo.height) for photo in photos] == [(40, 30), (20, 10)]
    assert {photo.processing_status for photo in photos} == {"imported"}
    assert photos[0].sha256 == calculate_sha256(first_image)
    assert photos[0].thumbnail_path is not None
    assert Path(photos[0].thumbnail_path).exists()
    assert photos[0].blur_score is not None


def test_import_folder_updates_existing_photos_without_duplicates(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "photos"
    image_path = image_folder / "image.jpg"
    create_image(image_path, (10, 10))
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    service = create_import_service(session_factory, tmp_path)

    first_summary = service.import_folder(image_folder)
    create_image(image_path, (12, 8))
    second_summary = service.import_folder(image_folder)

    with session_factory() as session:
        photos = PhotoRepository(session).list_photos()

    assert first_summary.files_added == 1
    assert second_summary.files_added == 0
    assert len(photos) == 1
    assert photos[0].width == 12
    assert photos[0].height == 8
    assert photos[0].blur_score is not None


def test_import_folder_generates_and_caches_image_embeddings(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "photos"
    first_image = image_folder / "first.jpg"
    second_image = image_folder / "second.png"
    create_image(first_image, (10, 10))
    create_image(second_image, (10, 10))
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    embedding_provider = FakeEmbeddingProvider(batch_size=1)
    service = create_import_service(
        session_factory,
        tmp_path,
        embedding_provider=embedding_provider,
    )

    service.import_folder(image_folder)

    with session_factory() as session:
        repository = PhotoRepository(session)
        photos = repository.list_photos()
        cached_embeddings = [
            repository.get_cached_photo_embedding(
                photo.id,
                model_name=embedding_provider.model_name,
                model_config=embedding_provider.pretrained,
            )
            for photo in photos
        ]

    assert embedding_provider.image_batches == [
        [first_image],
        [second_image],
    ]
    assert all(embedding is not None for embedding in cached_embeddings)
    assert {
        embedding.vector_dimension
        for embedding in cached_embeddings
        if embedding is not None
    } == {2}

    embedding_provider.image_batches.clear()

    service.import_folder(image_folder)

    assert embedding_provider.image_batches == []


def test_import_folder_records_corrupted_images_without_stopping(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "photos"
    create_image(image_folder / "good.jpg", (10, 20))
    bad_image = image_folder / "bad.jpg"
    bad_image.write_bytes(b"not an image")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    summary = create_import_service(session_factory, tmp_path).import_folder(
        image_folder,
    )

    with session_factory() as session:
        photos = PhotoRepository(session).list_photos()

    assert summary.files_seen == 2
    assert summary.files_imported == 2
    assert len(summary.errors) == 1
    assert len(photos) == 2
    assert {photo.processing_status for photo in photos} == {
        "imported",
        "metadata_error",
    }


def create_import_service(
    session_factory,
    tmp_path: Path,
    embedding_provider: FakeEmbeddingProvider | None = None,
) -> PhotoImportService:
    return PhotoImportService(
        session_factory,
        thumbnail_generator=ThumbnailGenerator(tmp_path / "thumb-cache"),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
    )


class FakeEmbeddingProvider:
    model_name = "ViT-B-32"
    pretrained = "test-pretrained"

    def __init__(self, batch_size: int = 2) -> None:
        self.batch_size = batch_size
        self.image_batches: list[list[Path]] = []

    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
        self.image_batches.append(paths)
        return [
            EmbeddingResult(
                vector=(0.6, 0.8),
                dimension=2,
                model_name=self.model_name,
                pretrained=self.pretrained,
                device="cpu",
            )
            for _path in paths
        ]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(
                vector=(0.6, 0.8),
                dimension=2,
                model_name=self.model_name,
                pretrained=self.pretrained,
                device="cpu",
            )
            for _text in texts
        ]
