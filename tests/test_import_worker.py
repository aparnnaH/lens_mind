from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from lensmind.db.repository import PhotoRepository, initialize_sqlite  # noqa: E402
from lensmind.services.embeddings import EmbeddingResult  # noqa: E402
from lensmind.services.photo_import import PhotoImportService  # noqa: E402
from lensmind.services.thumbnail_generation import ThumbnailGenerator  # noqa: E402
from lensmind.ui.import_worker import PhotoImportWorker  # noqa: E402


@pytest.fixture
def app() -> QApplication:
    existing_app = QApplication.instance()
    if existing_app is not None:
        return existing_app

    return QApplication([])


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color="white").save(path)


def test_import_worker_emits_progress_and_finished_status(
    tmp_path: Path,
    app: QApplication,
) -> None:
    create_image(tmp_path / "photos" / "one.jpg")
    create_image(tmp_path / "photos" / "two.png")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    worker = PhotoImportWorker(
        import_service=PhotoImportService(
            session_factory,
            thumbnail_generator=ThumbnailGenerator(tmp_path / "thumb-cache"),
            embedding_provider=FakeEmbeddingProvider(),
        ),
        folder=tmp_path / "photos",
    )
    stages: list[str] = []
    filenames: list[str] = []
    completed_counts: list[int] = []
    total_counts: list[int] = []
    error_counts: list[int] = []
    finished_statuses: list[str] = []

    worker.stage_changed.connect(stages.append)
    worker.current_filename_changed.connect(filenames.append)
    worker.completed_count_changed.connect(completed_counts.append)
    worker.total_count_changed.connect(total_counts.append)
    worker.error_count_changed.connect(error_counts.append)
    worker.finished_status_changed.connect(finished_statuses.append)

    worker.run()
    app.processEvents()

    with session_factory() as session:
        photos = PhotoRepository(session).list_photos()

    assert "discovering" in stages
    assert "importing" in stages
    assert "embedding" in stages
    assert "recording" in stages
    assert stages[-1] == "completed"
    assert "one.jpg" in filenames
    assert "two.png" in filenames
    assert completed_counts[-1] == 2
    assert max(total_counts) == 2
    assert error_counts[-1] == 0
    assert finished_statuses == ["completed"]
    assert len(photos) == 2


def test_import_worker_supports_cancellation(
    tmp_path: Path,
    app: QApplication,
) -> None:
    create_image(tmp_path / "photos" / "one.jpg")
    create_image(tmp_path / "photos" / "two.jpg")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    worker = PhotoImportWorker(
        import_service=PhotoImportService(
            session_factory,
            thumbnail_generator=ThumbnailGenerator(tmp_path / "thumb-cache"),
            embedding_provider=FakeEmbeddingProvider(),
        ),
        folder=tmp_path / "photos",
    )
    finished_statuses: list[str] = []

    def cancel_after_first_completed(completed_count: int) -> None:
        if completed_count == 1:
            worker.cancel()

    worker.completed_count_changed.connect(cancel_after_first_completed)
    worker.finished_status_changed.connect(finished_statuses.append)

    worker.run()
    app.processEvents()

    with session_factory() as session:
        photos = PhotoRepository(session).list_photos()

    assert finished_statuses == ["cancelled"]
    assert len(photos) == 1


class FakeEmbeddingProvider:
    model_name = "ViT-B-32"
    pretrained = "test-pretrained"
    batch_size = 2

    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
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
