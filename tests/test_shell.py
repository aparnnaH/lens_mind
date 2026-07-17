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
from lensmind.services.duplicate_detection import (
    DuplicateDetectionService,  # noqa: E402
)
from lensmind.services.faiss_search import FaissSearchResult  # noqa: E402
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


def test_top_search_bar_displays_semantic_results_and_clear_search(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        first = repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "first.jpg"),
                filename="first.jpg",
                file_size=100,
            ),
        )
        second = repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "second.jpg"),
                filename="second.jpg",
                file_size=200,
            ),
        )
    fake_search_service = FakeSemanticSearchService(
        [
            FaissSearchResult(photo_id=second.id, score=0.91),
            FaissSearchResult(photo_id=first.id, score=0.42),
        ],
    )
    monkeypatch.setattr(
        shell,
        "FaissPhotoSearchService",
        lambda *_args, **_kwargs: fake_search_service,
    )
    window = shell.MainWindow(session_factory)

    window.search_input.setText("beach sunset")
    window.search_button.click()
    app.processEvents()

    assert fake_search_service.queries == [("beach sunset", shell.SEARCH_RESULT_LIMIT)]
    assert [
        label.text()
        for label in window._all_photos_page.findChildren(QLabel, "photoFilename")
    ] == ["second.jpg", "first.jpg"]
    assert [
        label.text()
        for label in window._all_photos_page.findChildren(
            QLabel,
            "photoSimilarityScore",
        )
        if not label.isHidden()
    ] == ["Similarity 0.91", "Similarity 0.42"]
    assert window.clear_search_button.isEnabled()

    window.clear_search_button.click()
    app.processEvents()

    assert [
        label.text()
        for label in window._all_photos_page.findChildren(QLabel, "photoFilename")
    ] == ["first.jpg", "second.jpg"]
    assert [
        label.text()
        for label in window._all_photos_page.findChildren(
            QLabel,
            "photoSimilarityScore",
        )
        if not label.isHidden()
    ] == []
    assert window.search_input.text() == ""
    assert not window.clear_search_button.isEnabled()


def test_blurry_photos_page_reuses_gallery_with_blur_badges(
    app: QApplication,
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "blurry.jpg"),
                filename="blurry.jpg",
                file_size=100,
                blur_score=25.0,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "sharp.jpg"),
                filename="sharp.jpg",
                file_size=100,
                blur_score=250.0,
            ),
        )

    page = shell.AllPhotosPage(
        lambda: session_factory,
        title="Blurry Photos",
        empty_text="No blurry photos found",
        photos_loader=lambda repository: repository.list_blurry_photos(100.0),
    )

    page.load_photos()
    app.processEvents()

    assert page.findChild(QLabel, "pageTitle").text() == "Blurry Photos"
    assert [
        label.text()
        for label in page.findChildren(QLabel, "photoFilename")
    ] == ["blurry.jpg"]
    assert [
        label.text()
        for label in page.findChildren(QLabel, "photoBlurBadge")
    ] == ["Blur 25"]


def test_gallery_selection_updates_photo_details_inspector(
    app: QApplication,
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        source_folder = repository.add_source_folder(str(tmp_path / "photos"))
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "photos" / "selected.jpg"),
                source_folder_id=source_folder.id,
                filename="selected.jpg",
                file_size=2048,
                capture_timestamp=datetime(2026, 2, 3, 4, 5, 6),
                timestamp_source="exif",
                width=4000,
                height=3000,
                camera_make="Canon",
                camera_model="EOS",
                latitude=43.65,
                longitude=-79.38,
                blur_score=0.42,
            ),
        )

    window = shell.MainWindow(session_factory)

    window._show_photo_details(photo.id)
    app.processEvents()

    assert window._inspector.findChild(QLabel, "inspectorFilename").text() == (
        "selected.jpg"
    )
    assert window._inspector.findChild(QLabel, "inspectorCaptureDate").text() == (
        "2026-02-03"
    )
    assert window._inspector.findChild(QLabel, "inspectorTimestampSource").text() == (
        "exif"
    )
    assert window._inspector.findChild(QLabel, "inspectorDimensions").text() == (
        "4000 x 3000"
    )
    assert window._inspector.findChild(QLabel, "inspectorCameraDetails").text() == (
        "Canon EOS"
    )
    assert window._inspector.findChild(QLabel, "inspectorGpsCoordinates").text() == (
        "43.650000, -79.380000"
    )
    assert window._inspector.findChild(QLabel, "inspectorBlurScore").text() == "0.42"
    assert window._inspector.findChild(QLabel, "inspectorMissingFile").text() == "No"


def test_photo_details_inspector_shows_placeholders_for_missing_metadata(
    app: QApplication,
    tmp_path: Path,
) -> None:
    inspector = shell.PhotoDetailsInspector()
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        photo = PhotoRepository(session).add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "missing.jpg"),
                filename="missing.jpg",
                file_size=100,
                missing_file=True,
            ),
        )
        display_info = shell.PhotoDisplayService(session).get_photo_display_info(
            photo.id,
        )

    inspector.set_photo(display_info)

    assert inspector.findChild(QLabel, "inspectorPreviewPath").text() == "-"
    assert inspector.findChild(QLabel, "inspectorTimestampSource").text() == "-"
    assert inspector.findChild(QLabel, "inspectorDimensions").text() == "-"
    assert inspector.findChild(QLabel, "inspectorCameraDetails").text() == "-"
    assert inspector.findChild(QLabel, "inspectorGpsCoordinates").text() == "-"
    assert inspector.findChild(QLabel, "inspectorBlurScore").text() == "-"
    assert inspector.findChild(QLabel, "inspectorSourceFolder").text() == "-"
    assert inspector.findChild(QLabel, "inspectorMissingFile").text() == "Yes"
    assert not inspector.open_in_finder_button.isEnabled()
    assert not inspector.open_original_button.isEnabled()
    assert inspector.copy_path_button.isEnabled()


def test_photo_details_inspector_copies_original_path(
    app: QApplication,
    tmp_path: Path,
) -> None:
    inspector = shell.PhotoDetailsInspector()
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    original_path = tmp_path / "photo.jpg"
    with session_factory() as session:
        photo = PhotoRepository(session).add_or_update_photo(
            PhotoData(
                original_path=str(original_path),
                filename="photo.jpg",
                file_size=100,
            ),
        )
        display_info = shell.PhotoDisplayService(session).get_photo_display_info(
            photo.id,
        )

    inspector.set_photo(display_info)
    inspector.copy_path_button.click()

    assert QApplication.clipboard().text() == str(original_path)


def test_duplicates_page_shows_group_navigation_and_side_by_side_details(
    app: QApplication,
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    QImage(12, 10, QImage.Format.Format_RGB32).save(str(first_path))
    QImage(12, 10, QImage.Format.Format_RGB32).save(str(second_path))

    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(first_path),
                filename="first.png",
                file_size=1024,
                sha256="a" * 64,
                width=12,
                height=10,
                blur_score=12.5,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(second_path),
                filename="second.png",
                file_size=2048,
                sha256="a" * 64,
                width=12,
                height=10,
                blur_score=18.5,
            ),
        )
        DuplicateDetectionService(session).rebuild_duplicate_groups()

    page = shell.DuplicatesPage(lambda: session_factory)
    page.load_duplicate_groups()
    app.processEvents()

    assert page.group_list.count() == 1
    assert page.group_list.item(0).text() == "Group 1 - Exact duplicate"
    assert page.status_label.text() == "Exact duplicate"
    assert page.left_preview.filename_label.text() == "first.png"
    assert page.right_preview.filename_label.text() == "second.png"
    assert page.left_preview.dimensions_label.text() == "12 x 10"
    assert page.right_preview.file_size_label.text() == "2.0 KB"
    assert page.left_preview.blur_score_label.text() == "12.50"


def test_duplicates_page_actions_mark_reviewed_and_select_preferred(
    app: QApplication,
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        first = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/first.jpg",
                filename="first.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )
        second = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/second.jpg",
                filename="second.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )
        DuplicateDetectionService(session).rebuild_duplicate_groups()

    page = shell.DuplicatesPage(lambda: session_factory)
    page.load_duplicate_groups()
    app.processEvents()
    page._select_photo(page._selected_group.photos[1])
    page.select_preferred_button.click()

    with session_factory() as session:
        group = PhotoRepository(session).list_duplicate_groups()[0]

    assert first.id != second.id
    assert group.reviewed is True
    assert group.preferred_photo_id == second.id

    page.keep_all_button.click()
    with session_factory() as session:
        group = PhotoRepository(session).list_duplicate_groups()[0]

    assert group.reviewed is True
    assert group.preferred_photo_id is None


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
        blur_score=42.0,
    )

    item = shell.PhotoGridItem(photo, loader)

    assert item.thumbnail_state_label.text() == "Loading..."
    assert item.findChild(QLabel, "photoBlurBadge").text() == "Blur 42"
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


class FakeSemanticSearchService:
    def __init__(self, results: list[FaissSearchResult]) -> None:
        self._results = results
        self.queries: list[tuple[str, int]] = []

    def search_photos(self, query: str, limit: int) -> list[FaissSearchResult]:
        self.queries.append((query, limit))
        return self._results
