from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from lensmind.db.models import Photo  # noqa: E402
from lensmind.db.repository import (  # noqa: E402
    PhotoData,
    PhotoRepository,
    initialize_sqlite,
)
from lensmind.ui import shell  # noqa: E402
from lensmind.ui.thumbnail_loader import ThumbnailLoadResult  # noqa: E402


@pytest.fixture
def app() -> QApplication:
    existing_app = QApplication.instance()
    if existing_app is not None:
        return existing_app

    return QApplication([])


def test_run_application_starts_main_window(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeApplication:
        def __init__(self, argv: list[str]) -> None:
            assert argv == []
            events.append("app")

        def exec(self) -> int:
            events.append("exec")
            return 0

    class FakeMainWindow:
        def __init__(self) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    monkeypatch.setattr(shell, "QApplication", FakeApplication)
    monkeypatch.setattr(shell, "MainWindow", FakeMainWindow)

    assert shell.run_application([]) == 0
    assert events == ["app", "window", "show", "exec"]


def test_main_window_creates_navigation_pages(
    app: QApplication,
    tmp_path: Path,
) -> None:
    window = shell.MainWindow(initialize_sqlite(tmp_path / "lensmind.db"))
    app.processEvents()

    assert window.minimumWidth() == 1100
    assert window.minimumHeight() == 700
    assert window._sidebar.count() == len(shell.PAGE_TITLES)
    assert window._pages.count() == len(shell.PAGE_TITLES)

    for index, title in enumerate(shell.PAGE_TITLES):
        assert window._sidebar.item(index).text() == title

        page = window._pages.widget(index)
        page_title = page.findChild(QLabel, "pageTitle")
        assert page_title is not None
        assert page_title.text() == title

    target_index = shell.PAGE_TITLES.index("Evaluations")
    window._sidebar.setCurrentRow(target_index)

    assert window._pages.currentIndex() == target_index


def test_indexing_page_displays_import_progress(app: QApplication, tmp_path) -> None:
    page = shell.IndexingPage()

    page.prepare_for_import(tmp_path / "photos")
    page.set_stage("importing")
    page.set_current_filename("image.jpg")
    page.set_total_count(4)
    page.set_completed_count(2)
    page.set_error_count(1)

    assert page.source_folder_label.text() == str(tmp_path / "photos")
    assert page.stage_label.text() == "importing"
    assert page.current_filename_label.text() == "image.jpg"
    assert page.counts_label.text() == "2 / 4"
    assert page.error_count_label.text() == "1"
    assert page.progress_bar.value() == 2
    assert page.progress_bar.maximum() == 4
    assert page.cancel_button.isEnabled()


def test_import_folder_action_uses_native_folder_picker(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = shell.MainWindow(initialize_sqlite(tmp_path / "lensmind.db"))
    selected_folders: list[Path] = []

    monkeypatch.setattr(
        shell.QFileDialog,
        "getExistingDirectory",
        lambda *_args: str(tmp_path),
    )
    monkeypatch.setattr(window, "_start_import", selected_folders.append)

    assert window._import_folder_action.text() == "Import Folder"

    window._choose_import_folder()

    assert selected_folders == [tmp_path]


def test_all_photos_page_loads_repository_records(
    app: QApplication,
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "first.jpg"),
                filename="first.jpg",
                file_size=100,
                capture_timestamp=datetime(2026, 1, 2, 3, 4, 5),
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "second.jpg"),
                filename="second.jpg",
                file_size=200,
            ),
        )

    page = shell.AllPhotosPage(lambda: session_factory)

    page.load_photos()
    app.processEvents()

    filenames = [
        label.text()
        for label in page.findChildren(QLabel, "photoFilename")
    ]
    capture_dates = [
        label.text()
        for label in page.findChildren(QLabel, "photoCaptureDate")
    ]

    assert filenames == ["first.jpg", "second.jpg"]
    assert capture_dates == ["2026-01-02", "Unknown date"]
    assert page.findChild(QLabel, "emptyAllPhotosLabel").isHidden()


def test_photo_grid_item_shows_placeholder_while_thumbnail_loads(
    app: QApplication,
    tmp_path: Path,
) -> None:
    loader = FakeThumbnailLoader()
    photo = Photo(
        original_path=str(tmp_path / "photo.jpg"),
        filename="photo.jpg",
        file_size=100,
        thumbnail_path=str(tmp_path / "thumb.png"),
    )

    item = shell.PhotoGridItem(photo, loader)

    assert item.thumbnail_state_label.text() == "Loading..."
    assert loader.requested_path == tmp_path / "thumb.png"


def test_photo_grid_item_shows_missing_file_state(
    app: QApplication,
    tmp_path: Path,
) -> None:
    loader = FakeThumbnailLoader()
    photo = Photo(
        original_path=str(tmp_path / "missing.jpg"),
        filename="missing.jpg",
        file_size=100,
        thumbnail_path=str(tmp_path / "thumb.png"),
        missing_file=True,
    )

    item = shell.PhotoGridItem(photo, loader)

    assert item.thumbnail_state_label.text() == "Missing file"
    assert loader.requested_path is None


def test_photo_grid_item_handles_thumbnail_error(
    app: QApplication,
    tmp_path: Path,
) -> None:
    loader = FakeThumbnailLoader()
    thumbnail_path = tmp_path / "bad-thumb.png"
    photo = Photo(
        original_path=str(tmp_path / "photo.jpg"),
        filename="photo.jpg",
        file_size=100,
        thumbnail_path=str(thumbnail_path),
    )
    item = shell.PhotoGridItem(photo, loader)

    loader.callback(
        ThumbnailLoadResult(
            path=thumbnail_path,
            status="error",
            error="bad image",
        ),
    )

    assert item.thumbnail_state_label.text() == "Thumbnail error"


def test_photo_grid_item_applies_loaded_thumbnail(
    app: QApplication,
    tmp_path: Path,
) -> None:
    loader = FakeThumbnailLoader()
    thumbnail_path = tmp_path / "thumb.png"
    photo = Photo(
        original_path=str(tmp_path / "photo.jpg"),
        filename="photo.jpg",
        file_size=100,
        thumbnail_path=str(thumbnail_path),
    )
    item = shell.PhotoGridItem(photo, loader)

    loader.callback(
        ThumbnailLoadResult(
            path=thumbnail_path,
            status="loaded",
            image=QImage(10, 10, QImage.Format.Format_RGB32),
        ),
    )

    assert item.thumbnail_state_label.text() == ""
    assert item.thumbnail_label.pixmap() is not None


class FakeThumbnailLoader:
    def __init__(self) -> None:
        self.requested_path: Path | None = None
        self.callback = None

    def load(self, path, _size, callback):
        self.requested_path = Path(path)
        self.callback = callback
        return object()
