from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.models import Photo
from lensmind.db.repository import (
    AlbumData,
    DuplicateGroupData,
    DuplicatePhotoData,
    EvaluationRunData,
    PhotoRepository,
    TripData,
    initialize_sqlite,
)
from lensmind.services.blur_analysis import BlurThresholds
from lensmind.services.faiss_search import (
    FaissIndexError,
    FaissPhotoSearchService,
)
from lensmind.services.openclip_embeddings import (
    DEFAULT_OPENCLIP_MODEL,
    DEFAULT_OPENCLIP_PRETRAINED,
)
from lensmind.services.photo_display import PhotoDisplayInfo, PhotoDisplayService
from lensmind.services.photo_import import PhotoImportService
from lensmind.services.photo_ranking import PhotoRank, PhotoRankingService
from lensmind.services.trip_suggestions import SuggestedTrip, TripSuggestionService
from lensmind.ui.import_worker import PhotoImportWorker
from lensmind.ui.thumbnail_loader import ThumbnailLoader, ThumbnailLoadResult

PAGE_TITLES = (
    "All Photos",
    "Trips",
    "Albums",
    "Best Photos",
    "Duplicates",
    "Blurry Photos",
    "Indexing",
    "Evaluations",
    "Settings",
)
SEARCH_RESULT_LIMIT = 100


@dataclass(frozen=True)
class RankedPhotoResult:
    photo: Photo
    similarity_score: float


@dataclass(frozen=True)
class SidebarEntry:
    title: str
    page_title: str
    album_id: int | None = None


@dataclass(frozen=True)
class TripPageItem:
    kind: str
    name: str
    photo_ids: tuple[int, ...]
    trip_id: int | None = None
    cover_photo_id: int | None = None


class MainWindow(QMainWindow):
    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        super().__init__()

        self.setWindowTitle("LensMind")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1280, 800)

        self._session_factory = session_factory
        self._import_thread: QThread | None = None
        self._import_worker: PhotoImportWorker | None = None
        self._selected_photo_ids: set[int] = set()
        self._all_photos_page = AllPhotosPage(
            self._get_session_factory,
            title="All Photos",
            empty_text="No photos imported yet",
        )
        self._all_photos_page.photo_selected.connect(self._show_photo_details)
        self._blurry_photos_page = AllPhotosPage(
            self._get_session_factory,
            title="Blurry Photos",
            empty_text="No blurry photos found",
            photos_loader=lambda repository: repository.list_blurry_photos(
                BlurThresholds().blurry,
            ),
        )
        self._blurry_photos_page.photo_selected.connect(self._show_photo_details)
        self._trips_page = TripsPage(self._get_session_factory)
        self._trips_page.photo_selected.connect(self._show_photo_details)
        self._albums_page = AlbumsPage(
            self._get_session_factory,
            selected_photo_ids_provider=lambda: set(self._selected_photo_ids),
            albums_changed_callback=self._refresh_album_sidebar,
        )
        self._albums_page.photo_selected.connect(self._show_photo_details)
        self._best_photos_page = BestPhotosPage(self._get_session_factory)
        self._best_photos_page.photo_selected.connect(self._show_best_photo_details)
        self._duplicates_page = DuplicatesPage(self._get_session_factory)
        self._evaluations_page = EvaluationResultsPage(self._get_session_factory)
        self._indexing_page = IndexingPage()
        self._pages = QStackedWidget()
        self._sidebar_entries: list[SidebarEntry] = []
        self._sidebar = self._build_sidebar()
        self._inspector = self._build_inspector()

        self._build_toolbar()
        self._build_pages()
        self._build_layout()
        self._refresh_album_sidebar()

        self._sidebar.setCurrentRow(0)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.addWidget(QLabel("LensMind"))

        self._import_folder_action = QAction("Import Folder", self)
        self._import_folder_action.triggered.connect(self._choose_import_folder)
        toolbar.addAction(self._import_folder_action)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("semanticSearchInput")
        self.search_input.setPlaceholderText("Search photos")
        self.search_input.setMinimumWidth(260)
        self.search_input.returnPressed.connect(self._run_semantic_search)
        toolbar.addWidget(self.search_input)

        self.search_button = QPushButton("Search")
        self.search_button.setObjectName("semanticSearchButton")
        self.search_button.clicked.connect(self._run_semantic_search)
        toolbar.addWidget(self.search_button)

        self.clear_search_button = QPushButton("Clear Search")
        self.clear_search_button.setObjectName("clearSearchButton")
        self.clear_search_button.setEnabled(False)
        self.clear_search_button.clicked.connect(self._clear_search)
        toolbar.addWidget(self.clear_search_button)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_sidebar(self) -> QListWidget:
        sidebar = QListWidget()
        sidebar.setFixedWidth(220)
        sidebar.setSpacing(2)
        sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sidebar.currentRowChanged.connect(self._handle_sidebar_changed)

        return sidebar

    def _build_pages(self) -> None:
        for title in PAGE_TITLES:
            if title == "All Photos":
                self._pages.addWidget(self._all_photos_page)
            elif title == "Trips":
                self._pages.addWidget(self._trips_page)
            elif title == "Albums":
                self._pages.addWidget(self._albums_page)
            elif title == "Best Photos":
                self._pages.addWidget(self._best_photos_page)
            elif title == "Blurry Photos":
                self._pages.addWidget(self._blurry_photos_page)
            elif title == "Duplicates":
                self._pages.addWidget(self._duplicates_page)
            elif title == "Indexing":
                self._pages.addWidget(self._indexing_page)
            elif title == "Evaluations":
                self._pages.addWidget(self._evaluations_page)
            else:
                self._pages.addWidget(PlaceholderPage(title))

    def _build_inspector(self) -> PhotoDetailsInspector:
        return PhotoDetailsInspector()

    def _build_layout(self) -> None:
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._sidebar)
        layout.addWidget(self._pages, 1)
        layout.addWidget(self._inspector)
        self.setCentralWidget(content)

    def _choose_import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import Folder")
        if not folder:
            return

        self._start_import(Path(folder))

    def _start_import(self, folder: Path) -> None:
        if self._import_thread is not None:
            return

        indexing_index = PAGE_TITLES.index("Indexing")
        self._sidebar.setCurrentRow(indexing_index)
        self._indexing_page.prepare_for_import(folder)
        self._import_folder_action.setEnabled(False)

        import_service = PhotoImportService(self._get_session_factory())
        worker = PhotoImportWorker(import_service=import_service, folder=folder)
        thread = QThread(self)
        worker.moveToThread(thread)

        self._import_worker = worker
        self._import_thread = thread

        thread.started.connect(worker.run)
        worker.stage_changed.connect(self._indexing_page.set_stage)
        worker.current_filename_changed.connect(self._indexing_page.set_current_filename)
        worker.completed_count_changed.connect(self._indexing_page.set_completed_count)
        worker.total_count_changed.connect(self._indexing_page.set_total_count)
        worker.error_count_changed.connect(self._indexing_page.set_error_count)
        worker.finished_status_changed.connect(self._indexing_page.set_finished_status)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_import_worker)
        thread.finished.connect(self._all_photos_page.load_photos)
        thread.finished.connect(self._blurry_photos_page.load_photos)
        self._indexing_page.cancel_button.clicked.connect(worker.cancel)

        thread.start()

    def _handle_sidebar_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sidebar_entries):
            return

        entry = self._sidebar_entries[row]
        self._pages.setCurrentIndex(PAGE_TITLES.index(entry.page_title))
        if entry.page_title == "All Photos":
            self._all_photos_page.load_photos()
        elif entry.page_title == "Trips":
            self._trips_page.load_trips()
        elif entry.page_title == "Albums":
            self._albums_page.load_album(entry.album_id)
        elif entry.page_title == "Best Photos":
            self._best_photos_page.load_recommendations()
        elif entry.page_title == "Blurry Photos":
            self._blurry_photos_page.load_photos()
        elif entry.page_title == "Duplicates":
            self._duplicates_page.load_duplicate_groups()
        elif entry.page_title == "Evaluations":
            self._evaluations_page.load_runs()

    def _refresh_album_sidebar(self) -> None:
        current_entry = self._sidebar_entries[self._sidebar.currentRow()] if (
            0 <= self._sidebar.currentRow() < len(self._sidebar_entries)
        ) else None
        with self._get_session_factory()() as session:
            albums = PhotoRepository(session).list_albums()

        entries: list[SidebarEntry] = []
        for title in PAGE_TITLES:
            entries.append(SidebarEntry(title=title, page_title=title))
            if title == "Albums":
                entries.extend(
                    SidebarEntry(
                        title=f"  {album.name}",
                        page_title="Albums",
                        album_id=album.id,
                    )
                    for album in albums
                )

        self._sidebar.blockSignals(True)
        self._sidebar.clear()
        self._sidebar_entries = entries
        for entry in self._sidebar_entries:
            item = QListWidgetItem(entry.title)
            item.setSizeHint(QSize(180, 36))
            self._sidebar.addItem(item)
        self._sidebar.blockSignals(False)

        row = self._sidebar_row_for_entry(current_entry)
        self._sidebar.setCurrentRow(row)

    def _sidebar_row_for_entry(self, entry: SidebarEntry | None) -> int:
        if entry is None:
            return 0
        for index, candidate in enumerate(self._sidebar_entries):
            if (
                candidate.page_title == entry.page_title
                and candidate.album_id == entry.album_id
            ):
                return index
        return PAGE_TITLES.index(entry.page_title)

    def _clear_import_worker(self) -> None:
        self._indexing_page.cancel_button.clicked.disconnect()
        self._indexing_page.set_running(False)
        self._import_folder_action.setEnabled(True)
        self._import_worker = None
        self._import_thread = None

    def _show_photo_details(self, photo_id: int) -> None:
        self._selected_photo_ids.add(photo_id)
        with self._get_session_factory()() as session:
            display_info = PhotoDisplayService(session).get_photo_display_info(photo_id)

        self._inspector.set_photo(display_info)

    def _show_best_photo_details(self, photo_id: int) -> None:
        self._show_photo_details(photo_id)
        self._inspector.set_recommendation_reasons(
            self._best_photos_page.reasons_for_photo(photo_id),
        )

    def _run_semantic_search(self) -> None:
        query = self.search_input.text()
        try:
            search_results = FaissPhotoSearchService(
                self._get_session_factory(),
                _default_faiss_index_dir(),
                model_name=DEFAULT_OPENCLIP_MODEL,
                model_config=DEFAULT_OPENCLIP_PRETRAINED,
            ).search_photos(query, SEARCH_RESULT_LIMIT)
        except ValueError as error:
            self._show_all_photos_page()
            self._all_photos_page.show_search_error(str(error))
            self.clear_search_button.setEnabled(True)
            return
        except FaissIndexError as error:
            self._show_all_photos_page()
            self._all_photos_page.show_search_error(str(error))
            self.clear_search_button.setEnabled(True)
            return

        with self._get_session_factory()() as session:
            photos_by_id = {
                photo.id: photo
                for photo in PhotoRepository(session).list_photos()
            }

        ranked_photos = [
            RankedPhotoResult(
                photo=photos_by_id[result.photo_id],
                similarity_score=result.score,
            )
            for result in search_results
            if result.photo_id in photos_by_id
        ]
        self._show_all_photos_page()
        self._all_photos_page.show_search_results(query, ranked_photos)
        self.clear_search_button.setEnabled(True)

    def _clear_search(self) -> None:
        self.search_input.clear()
        self._all_photos_page.load_photos()
        self._show_all_photos_page()
        self.clear_search_button.setEnabled(False)

    def _show_all_photos_page(self) -> None:
        all_photos_index = PAGE_TITLES.index("All Photos")
        if self._sidebar.currentRow() != all_photos_index:
            self._sidebar.setCurrentRow(all_photos_index)

    def _get_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = initialize_sqlite(_default_database_path())

        return self._session_factory


class AllPhotosPage(QWidget):
    photo_selected = Signal(int)

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
        *,
        title: str = "All Photos",
        empty_text: str = "No photos imported yet",
        photos_loader: Callable[[PhotoRepository], list[Photo]] | None = None,
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._photos_loader = photos_loader or (
            lambda repository: repository.list_photos()
        )
        self._thumbnail_loader = ThumbnailLoader()
        self._empty_text = empty_text
        self._selected_photo_ids: set[int] = set()

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")

        self._empty_label = QLabel(empty_text)
        self._empty_label.setObjectName("emptyAllPhotosLabel")

        self._grid_container = QWidget()
        self._grid = QGridLayout(self._grid_container)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(16)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._grid_container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title_label)
        layout.addWidget(self._empty_label)
        layout.addWidget(scroll_area, 1)

    def load_photos(self) -> None:
        self._clear_grid()
        self._selected_photo_ids.clear()
        self._empty_label.setText(self._empty_text)
        session_factory = self._session_factory_provider()
        with session_factory() as session:
            photos = self._photos_loader(PhotoRepository(session))

        self._empty_label.setVisible(not photos)
        for index, photo in enumerate(photos):
            row, column = divmod(index, 4)
            item = PhotoGridItem(photo, self._thumbnail_loader)
            item.selected.connect(self._handle_photo_selected)
            self._grid.addWidget(item, row, column)

        self._grid.setRowStretch((len(photos) // 4) + 1, 1)

    def show_search_results(
        self,
        query: str,
        ranked_photos: list[RankedPhotoResult],
    ) -> None:
        self._clear_grid()
        self._selected_photo_ids.clear()
        self._empty_label.setText(f'No results for "{query.strip()}"')
        self._empty_label.setVisible(not ranked_photos)
        for index, result in enumerate(ranked_photos):
            row, column = divmod(index, 4)
            item = PhotoGridItem(
                result.photo,
                self._thumbnail_loader,
                similarity_score=result.similarity_score,
            )
            item.selected.connect(self._handle_photo_selected)
            self._grid.addWidget(item, row, column)

        self._grid.setRowStretch((len(ranked_photos) // 4) + 1, 1)

    def show_search_error(self, message: str) -> None:
        self._clear_grid()
        self._selected_photo_ids.clear()
        self._empty_label.setText(message)
        self._empty_label.setVisible(True)

    def selected_photo_ids(self) -> set[int]:
        return set(self._selected_photo_ids)

    def _handle_photo_selected(self, photo_id: int) -> None:
        self._selected_photo_ids.add(photo_id)
        self.photo_selected.emit(photo_id)

    def _clear_grid(self) -> None:
        while item := self._grid.takeAt(0):
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()


class PhotoGridItem(QFrame):
    selected = Signal(int)

    def __init__(
        self,
        photo: Photo,
        thumbnail_loader: ThumbnailLoader,
        *,
        similarity_score: float | None = None,
    ) -> None:
        super().__init__()
        self._photo_id = photo.id
        self.setObjectName("photoGridItem")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(180)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thumbnail_path = (
            Path(photo.thumbnail_path)
            if photo.thumbnail_path
            else Path(photo.original_path)
        )

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setObjectName("photoThumbnail")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setFixedSize(QSize(160, 120))
        self.thumbnail_state_label = QLabel()
        self.thumbnail_state_label.setObjectName("photoThumbnailState")
        self.thumbnail_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        filename_label = QLabel(photo.filename)
        filename_label.setObjectName("photoFilename")
        filename_label.setWordWrap(True)

        capture_date_label = QLabel(_format_capture_date(photo.capture_timestamp))
        capture_date_label.setObjectName("photoCaptureDate")
        blur_badge_label = QLabel(_format_blur_badge(photo.blur_score))
        blur_badge_label.setObjectName("photoBlurBadge")
        blur_badge_label.setVisible(photo.blur_score is not None)
        similarity_label = QLabel(_format_similarity_score(similarity_score))
        similarity_label.setObjectName("photoSimilarityScore")
        similarity_label.setVisible(similarity_score is not None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.thumbnail_label)
        layout.addWidget(self.thumbnail_state_label)
        layout.addWidget(similarity_label)
        layout.addWidget(blur_badge_label)
        layout.addWidget(filename_label)
        layout.addWidget(capture_date_label)

        self._load_thumbnail(photo, thumbnail_loader)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._photo_id is not None:
            self.selected.emit(self._photo_id)
        super().mousePressEvent(event)

    def _load_thumbnail(
        self,
        photo: Photo,
        thumbnail_loader: ThumbnailLoader,
    ) -> None:
        if photo.missing_file:
            self._set_thumbnail_state("Missing file", clear_image=True)
            return

        self._set_thumbnail_state("Loading...", clear_image=True)
        thumbnail_loader.load(
            self._thumbnail_path,
            self.thumbnail_label.size(),
            self._handle_thumbnail_loaded,
        )

    def _handle_thumbnail_loaded(self, result: ThumbnailLoadResult) -> None:
        if self._thumbnail_path is None or result.path != self._thumbnail_path:
            return

        if result.status == "missing":
            self._set_thumbnail_state("Missing thumbnail", clear_image=True)
            return

        if result.status == "error" or result.image is None:
            self._set_thumbnail_state("Thumbnail error", clear_image=True)
            return

        pixmap = QPixmap.fromImage(result.image)
        if pixmap.isNull():
            self._set_thumbnail_state("Thumbnail error", clear_image=True)
            return

        self.thumbnail_label.setPixmap(pixmap)
        self._set_thumbnail_state("")

    def _set_thumbnail_state(self, text: str, *, clear_image: bool = False) -> None:
        if clear_image:
            self.thumbnail_label.clear()
        self.thumbnail_state_label.setText(text)


class PhotoDetailsInspector(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("rightInspector")
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self._original_path: Path | None = None

        title_label = QLabel("Inspector")
        title_label.setObjectName("inspectorTitle")

        self._fields = {
            "filename": _detail_value_label("inspectorFilename"),
            "original_path": _detail_value_label("inspectorOriginalPath"),
            "file_size": _detail_value_label("inspectorFileSize"),
            "capture_date": _detail_value_label("inspectorCaptureDate"),
            "timestamp_source": _detail_value_label("inspectorTimestampSource"),
            "dimensions": _detail_value_label("inspectorDimensions"),
            "camera_details": _detail_value_label("inspectorCameraDetails"),
            "gps_coordinates": _detail_value_label("inspectorGpsCoordinates"),
            "blur_score": _detail_value_label("inspectorBlurScore"),
            "source_folder": _detail_value_label("inspectorSourceFolder"),
            "missing_file": _detail_value_label("inspectorMissingFile"),
            "preview_path": _detail_value_label("inspectorPreviewPath"),
            "recommendation_reasons": _detail_value_label(
                "inspectorRecommendationReasons",
            ),
        }

        self.open_in_finder_button = QPushButton("Open in Finder")
        self.open_in_finder_button.setObjectName("openInFinderButton")
        self.open_original_button = QPushButton("Open Original")
        self.open_original_button.setObjectName("openOriginalButton")
        self.copy_path_button = QPushButton("Copy Path")
        self.copy_path_button.setObjectName("copyPathButton")

        self.open_in_finder_button.clicked.connect(self._open_in_finder)
        self.open_original_button.clicked.connect(self._open_original)
        self.copy_path_button.clicked.connect(self._copy_path)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title_label)
        for label_text, key in (
            ("Filename", "filename"),
            ("Preview", "preview_path"),
            ("Original", "original_path"),
            ("File size", "file_size"),
            ("Capture date", "capture_date"),
            ("Timestamp", "timestamp_source"),
            ("Dimensions", "dimensions"),
            ("Camera", "camera_details"),
            ("GPS", "gps_coordinates"),
            ("Blur score", "blur_score"),
            ("Recommendation reasons", "recommendation_reasons"),
            ("Source folder", "source_folder"),
            ("Missing file", "missing_file"),
        ):
            layout.addWidget(QLabel(label_text))
            layout.addWidget(self._fields[key])

        layout.addWidget(self.open_in_finder_button)
        layout.addWidget(self.open_original_button)
        layout.addWidget(self.copy_path_button)
        layout.addStretch(1)
        self.set_photo(None)

    def set_photo(self, display_info: PhotoDisplayInfo | None) -> None:
        if display_info is None:
            self._original_path = None
            for label in self._fields.values():
                label.setText("Select a photo")
            self._set_actions_enabled(False)
            return

        self._original_path = display_info.original_path
        self._fields["filename"].setText(display_info.filename)
        self._fields["preview_path"].setText(_format_optional_path(display_info.preview_path))
        self._fields["original_path"].setText(str(display_info.original_path))
        self._fields["file_size"].setText(_format_file_size(display_info.file_size))
        self._fields["capture_date"].setText(_format_capture_date(display_info.capture_date))
        self._fields["timestamp_source"].setText(
            _format_optional_text(display_info.timestamp_source),
        )
        self._fields["dimensions"].setText(_format_dimensions(display_info.dimensions))
        self._fields["camera_details"].setText(
            _format_optional_text(display_info.camera_details),
        )
        self._fields["gps_coordinates"].setText(
            _format_gps_coordinates(display_info.gps_coordinates),
        )
        self._fields["blur_score"].setText(_format_optional_number(display_info.blur_score))
        self._fields["recommendation_reasons"].setText("-")
        self._fields["source_folder"].setText(
            _format_optional_path(display_info.source_folder),
        )
        self._fields["missing_file"].setText(
            "Yes" if display_info.missing_file else "No",
        )
        self._set_actions_enabled(not display_info.missing_file)
        self.copy_path_button.setEnabled(True)

    def _open_in_finder(self) -> None:
        if self._original_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._original_path.parent)))

    def _open_original(self) -> None:
        if self._original_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._original_path)))

    def _copy_path(self) -> None:
        if self._original_path is None:
            return
        QApplication.clipboard().setText(str(self._original_path))

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.open_in_finder_button.setEnabled(enabled)
        self.open_original_button.setEnabled(enabled)
        self.copy_path_button.setEnabled(enabled)

    def set_recommendation_reasons(self, reasons: tuple[str, ...]) -> None:
        self._fields["recommendation_reasons"].setText(
            "\n".join(reasons) if reasons else "-",
        )


class BestPhotosPage(QWidget):
    photo_selected = Signal(int)

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._hidden_photo_ids: set[int] = set()
        self._ranks_by_photo_id: dict[int, PhotoRank] = {}
        self.gallery = AllPhotosPage(
            session_factory_provider,
            title="Best Photos",
            empty_text="No recommendations yet",
            photos_loader=self._load_ranked_photos,
        )
        self.gallery.photo_selected.connect(self.photo_selected)

        self.hide_recommendation_button = QPushButton("Hide Recommendation")
        self.hide_recommendation_button.setObjectName("hideRecommendationButton")
        self.hide_recommendation_button.clicked.connect(self._hide_selected)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.hide_recommendation_button)
        layout.addWidget(self.gallery, 1)

    def load_recommendations(self) -> None:
        self.gallery.load_photos()

    def reasons_for_photo(self, photo_id: int) -> tuple[str, ...]:
        rank = self._ranks_by_photo_id.get(photo_id)
        if rank is None:
            return ()
        return rank.reasons

    def _load_ranked_photos(self, repository: PhotoRepository) -> list[Photo]:
        photos_by_id = {
            photo.id: photo
            for photo in repository.list_photos()
        }
        ranks = PhotoRankingService(repository._session).rank_photos(
            list(photos_by_id.values()),
        )
        visible_ranks = [
            rank
            for rank in ranks
            if rank.photo_id not in self._hidden_photo_ids
        ]
        self._ranks_by_photo_id = {
            rank.photo_id: rank
            for rank in visible_ranks
        }
        return [
            photos_by_id[rank.photo_id]
            for rank in visible_ranks
            if rank.photo_id in photos_by_id
        ]

    def _hide_selected(self) -> None:
        selected_photo_ids = self.gallery.selected_photo_ids()
        if not selected_photo_ids:
            return

        self._hidden_photo_ids.update(selected_photo_ids)
        self.load_recommendations()


class TripsPage(QWidget):
    photo_selected = Signal(int)

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._items: list[TripPageItem] = []
        self._dismissed_suggestion_keys: set[tuple[int, ...]] = set()
        self._current_photo_ids: tuple[int, ...] = ()

        title_label = QLabel("Trips")
        title_label.setObjectName("pageTitle")
        self.status_label = QLabel("No trip suggestions")
        self.status_label.setObjectName("tripStatus")

        self.trip_list = QListWidget()
        self.trip_list.setObjectName("tripSuggestionList")
        self.trip_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.trip_list.currentRowChanged.connect(self._show_item)

        self.accept_button = QPushButton("Accept Suggestion")
        self.accept_button.setObjectName("acceptTripSuggestionButton")
        self.rename_button = QPushButton("Rename")
        self.rename_button.setObjectName("renameTripButton")
        self.merge_button = QPushButton("Merge")
        self.merge_button.setObjectName("mergeTripButton")
        self.remove_suggestion_button = QPushButton("Remove Suggestion")
        self.remove_suggestion_button.setObjectName("removeTripSuggestionButton")
        self.choose_cover_button = QPushButton("Choose Cover Photo")
        self.choose_cover_button.setObjectName("chooseTripCoverButton")

        self.accept_button.clicked.connect(self._accept_suggestions)
        self.rename_button.clicked.connect(self._rename_trip)
        self.merge_button.clicked.connect(self._merge_selected_items)
        self.remove_suggestion_button.clicked.connect(self._remove_suggestions)
        self.choose_cover_button.clicked.connect(self._choose_cover_photo)

        self.gallery = AllPhotosPage(
            session_factory_provider,
            title="Trip Photos",
            empty_text="No photos in this trip",
            photos_loader=self._load_current_photos,
        )
        self.gallery.photo_selected.connect(self.photo_selected)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.accept_button)
        action_layout.addWidget(self.rename_button)
        action_layout.addWidget(self.merge_button)
        action_layout.addWidget(self.remove_suggestion_button)
        action_layout.addWidget(self.choose_cover_button)
        action_layout.addStretch(1)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.trip_list, 1)
        content_layout.addWidget(self.gallery, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addLayout(action_layout)
        layout.addLayout(content_layout, 1)
        self._set_actions_enabled(False)

    def load_trips(self) -> None:
        current_key = self._current_item_key()
        with self._session_factory_provider()() as session:
            repository = PhotoRepository(session)
            trips = repository.list_trips()
            suggestion_service = TripSuggestionService(session)
            suggestions = suggestion_service.suggest_trips()
            saved_trip_photo_sets = {
                tuple(photo.id for photo in repository.list_trip_photos(trip.id))
                for trip in trips
            }

        self._items = [
            _trip_page_item_from_trip(trip, self._trip_photo_ids(trip.id))
            for trip in trips
        ]
        self._items.extend(
            _trip_page_item_from_suggestion(suggestion)
            for suggestion in suggestions
            if suggestion.photo_ids not in saved_trip_photo_sets
            and suggestion.photo_ids not in self._dismissed_suggestion_keys
        )
        self._reload_list(current_key)

    def _trip_photo_ids(self, trip_id: int) -> tuple[int, ...]:
        with self._session_factory_provider()() as session:
            return tuple(
                photo.id
                for photo in PhotoRepository(session).list_trip_photos(trip_id)
            )

    def _reload_list(
        self,
        selected_key: tuple[str, int | tuple[int, ...]] | None,
    ) -> None:
        self.trip_list.blockSignals(True)
        self.trip_list.clear()
        for item in self._items:
            list_item = QListWidgetItem(_format_trip_item(item))
            self.trip_list.addItem(list_item)
        self.trip_list.blockSignals(False)

        if not self._items:
            self._show_item(-1)
            return

        selected_row = 0
        if selected_key is not None:
            for index, item in enumerate(self._items):
                if _trip_item_key(item) == selected_key:
                    selected_row = index
                    break
        self.trip_list.setCurrentRow(selected_row)

    def _show_item(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            self._current_photo_ids = ()
            self.status_label.setText("No trip suggestions")
            self.gallery.show_search_error("No trip selected")
            self._set_actions_enabled(False)
            return

        item = self._items[row]
        self._current_photo_ids = item.photo_ids
        self.status_label.setText(_format_trip_status(item))
        self.gallery.load_photos()
        self._set_actions_enabled(True)

    def _load_current_photos(self, repository: PhotoRepository) -> list[Photo]:
        photos_by_id = {
            photo.id: photo
            for photo in repository.list_photos()
        }
        return [
            photos_by_id[photo_id]
            for photo_id in self._current_photo_ids
            if photo_id in photos_by_id
        ]

    def _accept_suggestions(self) -> None:
        selected_items = self._selected_items(kind="suggestion")
        if not selected_items:
            return

        with self._session_factory_provider()() as session:
            repository = PhotoRepository(session)
            for item in selected_items:
                repository.create_trip(item.name, list(item.photo_ids))
                self._dismissed_suggestion_keys.add(item.photo_ids)
        self.load_trips()

    def _rename_trip(self) -> None:
        item = self._current_item()
        if item is None or item.kind != "trip" or item.trip_id is None:
            return

        name, accepted = QInputDialog.getText(
            self,
            "Rename Trip",
            "Trip name",
            text=item.name,
        )
        if not accepted:
            return

        with self._session_factory_provider()() as session:
            PhotoRepository(session).rename_trip(item.trip_id, name)
        self.load_trips()

    def _merge_selected_items(self) -> None:
        selected_items = self._selected_items()
        if len(selected_items) < 2:
            return

        selected_photo_ids = _merged_photo_ids(selected_items)
        selected_trips = [
            item
            for item in selected_items
            if item.kind == "trip" and item.trip_id is not None
        ]
        with self._session_factory_provider()() as session:
            repository = PhotoRepository(session)
            if selected_trips:
                target_trip_id = selected_trips[0].trip_id
                assert target_trip_id is not None
                repository.add_photos_to_trip(target_trip_id, list(selected_photo_ids))
                repository.merge_trips(
                    target_trip_id,
                    [
                        item.trip_id
                        for item in selected_trips[1:]
                        if item.trip_id is not None
                    ],
                )
            else:
                repository.create_trip("Merged Trip", list(selected_photo_ids))

        for item in selected_items:
            if item.kind == "suggestion":
                self._dismissed_suggestion_keys.add(item.photo_ids)
        self.load_trips()

    def _remove_suggestions(self) -> None:
        suggestions = self._selected_items(kind="suggestion")
        if not suggestions:
            return

        for item in suggestions:
            self._dismissed_suggestion_keys.add(item.photo_ids)
        self.load_trips()

    def _choose_cover_photo(self) -> None:
        item = self._current_item()
        if item is None or item.kind != "trip" or item.trip_id is None:
            return

        selected_photo_ids = sorted(self.gallery.selected_photo_ids())
        if not selected_photo_ids:
            return

        with self._session_factory_provider()() as session:
            PhotoRepository(session).set_trip_cover(
                item.trip_id,
                selected_photo_ids[0],
            )
        self.load_trips()

    def _selected_items(self, kind: str | None = None) -> list[TripPageItem]:
        rows = sorted(index.row() for index in self.trip_list.selectedIndexes())
        items = [
            self._items[row]
            for row in rows
            if 0 <= row < len(self._items)
        ]
        if kind is not None:
            return [item for item in items if item.kind == kind]
        return items

    def _current_item(self) -> TripPageItem | None:
        row = self.trip_list.currentRow()
        if row < 0 or row >= len(self._items):
            return None
        return self._items[row]

    def _current_item_key(self) -> tuple[str, int | tuple[int, ...]] | None:
        item = self._current_item()
        if item is None:
            return None
        return _trip_item_key(item)

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.accept_button.setEnabled(enabled)
        self.rename_button.setEnabled(enabled)
        self.merge_button.setEnabled(enabled)
        self.remove_suggestion_button.setEnabled(enabled)
        self.choose_cover_button.setEnabled(enabled)


class AlbumsPage(QWidget):
    photo_selected = Signal(int)

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
        *,
        selected_photo_ids_provider: Callable[[], set[int]],
        albums_changed_callback: Callable[[], None],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._selected_photo_ids_provider = selected_photo_ids_provider
        self._albums_changed_callback = albums_changed_callback
        self._album_id: int | None = None
        self._albums: list[AlbumData] = []

        title_label = QLabel("Albums")
        title_label.setObjectName("pageTitle")
        self.album_name_label = QLabel("Choose or create an album")
        self.album_name_label.setObjectName("albumName")

        self.create_album_button = QPushButton("Create Album")
        self.create_album_button.setObjectName("createAlbumButton")
        self.rename_album_button = QPushButton("Rename Album")
        self.rename_album_button.setObjectName("renameAlbumButton")
        self.add_selected_button = QPushButton("Add Selected Photos")
        self.add_selected_button.setObjectName("addSelectedPhotosButton")
        self.remove_selected_button = QPushButton("Remove Photos")
        self.remove_selected_button.setObjectName("removeSelectedPhotosButton")
        self.choose_cover_button = QPushButton("Choose Cover Photo")
        self.choose_cover_button.setObjectName("chooseAlbumCoverButton")

        self.create_album_button.clicked.connect(self._create_album)
        self.rename_album_button.clicked.connect(self._rename_album)
        self.add_selected_button.clicked.connect(self._add_selected_photos)
        self.remove_selected_button.clicked.connect(self._remove_selected_photos)
        self.choose_cover_button.clicked.connect(self._choose_cover_photo)

        self.gallery = AllPhotosPage(
            session_factory_provider,
            title="Album Photos",
            empty_text="No photos in this album",
            photos_loader=self._load_album_photos,
        )
        self.gallery.photo_selected.connect(self.photo_selected)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.create_album_button)
        action_layout.addWidget(self.rename_album_button)
        action_layout.addWidget(self.add_selected_button)
        action_layout.addWidget(self.remove_selected_button)
        action_layout.addWidget(self.choose_cover_button)
        action_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title_label)
        layout.addWidget(self.album_name_label)
        layout.addLayout(action_layout)
        layout.addWidget(self.gallery, 1)
        self._set_album_actions_enabled(False)

    def load_album(self, album_id: int | None) -> None:
        self._album_id = album_id
        with self._session_factory_provider()() as session:
            self._albums = PhotoRepository(session).list_albums()

        album = self._selected_album()
        if album is None:
            self.album_name_label.setText("Choose or create an album")
            self.gallery.show_search_error("No album selected")
            self._set_album_actions_enabled(False)
            return

        self.album_name_label.setText(
            f"{album.name} - {album.photo_count} photos",
        )
        self._set_album_actions_enabled(True)
        self.gallery.load_photos()

    def _load_album_photos(self, repository: PhotoRepository) -> list[Photo]:
        if self._album_id is None:
            return []
        return repository.list_album_photos(self._album_id)

    def _create_album(self) -> None:
        name, accepted = QInputDialog.getText(self, "Create Album", "Album name")
        if not accepted:
            return

        with self._session_factory_provider()() as session:
            album = PhotoRepository(session).create_album(name)

        self._album_id = album.id
        self._albums_changed_callback()
        self.load_album(album.id)

    def _rename_album(self) -> None:
        album = self._selected_album()
        if album is None:
            return

        name, accepted = QInputDialog.getText(
            self,
            "Rename Album",
            "Album name",
            text=album.name,
        )
        if not accepted:
            return

        with self._session_factory_provider()() as session:
            PhotoRepository(session).rename_album(album.id, name)

        self._albums_changed_callback()
        self.load_album(album.id)

    def _add_selected_photos(self) -> None:
        if self._album_id is None:
            return
        album_id = self._album_id
        photo_ids = sorted(self._selected_photo_ids_provider())
        with self._session_factory_provider()() as session:
            PhotoRepository(session).add_photos_to_album(album_id, photo_ids)
        self._albums_changed_callback()
        self.load_album(album_id)

    def _remove_selected_photos(self) -> None:
        if self._album_id is None:
            return
        album_id = self._album_id
        photo_ids = sorted(self.gallery.selected_photo_ids())
        with self._session_factory_provider()() as session:
            PhotoRepository(session).remove_photos_from_album(album_id, photo_ids)
        self._albums_changed_callback()
        self.load_album(album_id)

    def _choose_cover_photo(self) -> None:
        if self._album_id is None:
            return
        album_id = self._album_id
        selected_ids = sorted(self.gallery.selected_photo_ids())
        if not selected_ids:
            return
        with self._session_factory_provider()() as session:
            PhotoRepository(session).set_album_cover(album_id, selected_ids[0])
        self._albums_changed_callback()
        self.load_album(album_id)

    def _selected_album(self) -> AlbumData | None:
        for album in self._albums:
            if album.id == self._album_id:
                return album
        return None

    def _set_album_actions_enabled(self, enabled: bool) -> None:
        self.rename_album_button.setEnabled(enabled)
        self.add_selected_button.setEnabled(enabled)
        self.remove_selected_button.setEnabled(enabled)
        self.choose_cover_button.setEnabled(enabled)


class DuplicatesPage(QWidget):
    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._groups: list[DuplicateGroupData] = []
        self._selected_group: DuplicateGroupData | None = None
        self._selected_photo: DuplicatePhotoData | None = None

        title_label = QLabel("Duplicates")
        title_label.setObjectName("pageTitle")

        self.group_list = QListWidget()
        self.group_list.setObjectName("duplicateGroupList")
        self.group_list.currentRowChanged.connect(self._show_group)

        self.status_label = QLabel("No duplicate groups")
        self.status_label.setObjectName("duplicateStatus")
        self.reviewed_label = QLabel("-")
        self.reviewed_label.setObjectName("duplicateReviewedStatus")
        self.preferred_label = QLabel("-")
        self.preferred_label.setObjectName("duplicatePreferredStatus")

        self.left_preview = DuplicatePreviewPanel("Photo A")
        self.right_preview = DuplicatePreviewPanel("Photo B")
        self.left_preview.selected.connect(self._select_photo)
        self.right_preview.selected.connect(self._select_photo)

        self.open_in_finder_button = QPushButton("Open in Finder")
        self.open_in_finder_button.setObjectName("duplicateOpenInFinderButton")
        self.keep_all_button = QPushButton("Keep All")
        self.keep_all_button.setObjectName("duplicateKeepAllButton")
        self.mark_reviewed_button = QPushButton("Mark Reviewed")
        self.mark_reviewed_button.setObjectName("duplicateMarkReviewedButton")
        self.select_preferred_button = QPushButton("Select Preferred")
        self.select_preferred_button.setObjectName("duplicateSelectPreferredButton")

        self.open_in_finder_button.clicked.connect(self._open_selected_in_finder)
        self.keep_all_button.clicked.connect(self._keep_all)
        self.mark_reviewed_button.clicked.connect(self._mark_reviewed)
        self.select_preferred_button.clicked.connect(self._select_preferred)

        previews_layout = QHBoxLayout()
        previews_layout.addWidget(self.left_preview)
        previews_layout.addWidget(self.right_preview)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.open_in_finder_button)
        action_layout.addWidget(self.keep_all_button)
        action_layout.addWidget(self.mark_reviewed_button)
        action_layout.addWidget(self.select_preferred_button)

        detail_layout = QVBoxLayout()
        detail_layout.addWidget(self.status_label)
        detail_layout.addWidget(self.reviewed_label)
        detail_layout.addWidget(self.preferred_label)
        detail_layout.addLayout(previews_layout)
        detail_layout.addLayout(action_layout)
        detail_layout.addStretch(1)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.group_list, 1)
        content_layout.addLayout(detail_layout, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title_label)
        layout.addLayout(content_layout, 1)
        self._set_actions_enabled(False)

    def load_duplicate_groups(self) -> None:
        current_group_id = (
            self._selected_group.id if self._selected_group is not None else None
        )
        with self._session_factory_provider()() as session:
            self._groups = PhotoRepository(session).list_duplicate_groups()

        self.group_list.clear()
        for index, duplicate_group in enumerate(self._groups, start=1):
            item = QListWidgetItem(
                f"Group {index} - {_format_duplicate_status(duplicate_group)}",
            )
            self.group_list.addItem(item)

        if not self._groups:
            self._show_group(-1)
            return

        selected_index = 0
        if current_group_id is not None:
            for index, duplicate_group in enumerate(self._groups):
                if duplicate_group.id == current_group_id:
                    selected_index = index
                    break
        self.group_list.setCurrentRow(selected_index)

    def _show_group(self, row: int) -> None:
        if row < 0 or row >= len(self._groups):
            self._selected_group = None
            self._selected_photo = None
            self.status_label.setText("No duplicate groups")
            self.reviewed_label.setText("-")
            self.preferred_label.setText("-")
            self.left_preview.set_photo(None, preferred_photo_id=None)
            self.right_preview.set_photo(None, preferred_photo_id=None)
            self._set_actions_enabled(False)
            return

        self._selected_group = self._groups[row]
        self._selected_photo = self._selected_group.photos[0]
        self.status_label.setText(_format_duplicate_status(self._selected_group))
        self.reviewed_label.setText(
            "Reviewed" if self._selected_group.reviewed else "Needs review",
        )
        self.preferred_label.setText(
            _format_preferred_photo(self._selected_group),
        )
        self.left_preview.set_photo(
            self._photo_at(0),
            preferred_photo_id=self._selected_group.preferred_photo_id,
        )
        self.right_preview.set_photo(
            self._photo_at(1),
            preferred_photo_id=self._selected_group.preferred_photo_id,
        )
        self.left_preview.set_selected(True)
        self.right_preview.set_selected(False)
        self._set_actions_enabled(True)

    def _photo_at(self, index: int) -> DuplicatePhotoData | None:
        if self._selected_group is None or index >= len(self._selected_group.photos):
            return None
        return self._selected_group.photos[index]

    def _select_photo(self, photo: DuplicatePhotoData) -> None:
        self._selected_photo = photo
        self.left_preview.set_selected(self.left_preview.photo_id == photo.id)
        self.right_preview.set_selected(self.right_preview.photo_id == photo.id)

    def _open_selected_in_finder(self) -> None:
        if self._selected_photo is None:
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(self._selected_photo.original_path).parent)),
        )

    def _keep_all(self) -> None:
        if self._selected_group is None:
            return
        with self._session_factory_provider()() as session:
            PhotoRepository(session).keep_all_duplicate_group_photos(
                self._selected_group.id,
            )
        self.load_duplicate_groups()

    def _mark_reviewed(self) -> None:
        if self._selected_group is None:
            return
        with self._session_factory_provider()() as session:
            PhotoRepository(session).mark_duplicate_group_reviewed(
                self._selected_group.id,
            )
        self.load_duplicate_groups()

    def _select_preferred(self) -> None:
        if self._selected_group is None or self._selected_photo is None:
            return
        with self._session_factory_provider()() as session:
            PhotoRepository(session).select_preferred_duplicate_photo(
                self._selected_group.id,
                self._selected_photo.id,
            )
        self.load_duplicate_groups()

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.open_in_finder_button.setEnabled(enabled)
        self.keep_all_button.setEnabled(enabled)
        self.mark_reviewed_button.setEnabled(enabled)
        self.select_preferred_button.setEnabled(enabled)


class DuplicatePreviewPanel(QFrame):
    selected = Signal(object)

    def __init__(self, title: str) -> None:
        super().__init__()
        self.photo_id: int | None = None
        self._photo: DuplicatePhotoData | None = None
        self.setObjectName("duplicatePreviewPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("duplicatePreviewTitle")
        self.preview_label = QLabel("No photo")
        self.preview_label.setObjectName("duplicatePreviewImage")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(QSize(240, 180))
        self.filename_label = QLabel("-")
        self.filename_label.setObjectName("duplicateFilename")
        self.dimensions_label = QLabel("-")
        self.dimensions_label.setObjectName("duplicateDimensions")
        self.file_size_label = QLabel("-")
        self.file_size_label.setObjectName("duplicateFileSize")
        self.blur_score_label = QLabel("-")
        self.blur_score_label.setObjectName("duplicateBlurScore")
        self.preferred_label = QLabel("")
        self.preferred_label.setObjectName("duplicatePreferredBadge")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.preferred_label)
        layout.addWidget(self.filename_label)
        layout.addWidget(self.dimensions_label)
        layout.addWidget(self.file_size_label)
        layout.addWidget(self.blur_score_label)

    def set_photo(
        self,
        photo: DuplicatePhotoData | None,
        *,
        preferred_photo_id: int | None,
    ) -> None:
        self._photo = photo
        self.photo_id = photo.id if photo is not None else None
        if photo is None:
            self.preview_label.clear()
            self.preview_label.setText("No photo")
            self.filename_label.setText("-")
            self.dimensions_label.setText("-")
            self.file_size_label.setText("-")
            self.blur_score_label.setText("-")
            self.preferred_label.setText("")
            return

        self._set_preview(photo)
        self.filename_label.setText(photo.filename)
        self.dimensions_label.setText(_format_duplicate_dimensions(photo))
        self.file_size_label.setText(_format_file_size(photo.file_size))
        self.blur_score_label.setText(_format_optional_number(photo.blur_score))
        self.preferred_label.setText(
            "Preferred" if preferred_photo_id == photo.id else "",
        )

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._photo is not None:
            self.selected.emit(self._photo)
        super().mousePressEvent(event)

    def _set_preview(self, photo: DuplicatePhotoData) -> None:
        preview_path = photo.thumbnail_path or photo.original_path
        if photo.missing_file:
            self.preview_label.clear()
            self.preview_label.setText("Missing file")
            return

        pixmap = QPixmap(preview_path)
        if pixmap.isNull():
            self.preview_label.clear()
            self.preview_label.setText("Preview unavailable")
            return

        self.preview_label.setText("")
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ),
        )


class EvaluationResultsPage(QWidget):
    METRIC_FIELDS = (
        ("precision", "Precision@K"),
        ("recall", "Recall@K"),
        ("mean_reciprocal_rank", "Mean Reciprocal Rank"),
        ("average_search_latency_ms", "Average Search Latency"),
        ("duplicate_precision", "Duplicate Precision"),
        ("duplicate_recall", "Duplicate Recall"),
        ("duplicate_f1", "Duplicate F1"),
        ("blur_precision", "Blur Precision"),
        ("blur_recall", "Blur Recall"),
        ("blur_f1", "Blur F1"),
    )

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._metric_labels: dict[str, QLabel] = {}

        title_label = QLabel("Evaluations")
        title_label.setObjectName("pageTitle")

        self.latest_run_date_label = QLabel("Unavailable")
        self.latest_run_date_label.setObjectName("latestEvaluationRunDate")

        metrics_layout = QGridLayout()
        metrics_layout.setHorizontalSpacing(24)
        metrics_layout.setVerticalSpacing(8)
        for row, (metric_key, label) in enumerate(self.METRIC_FIELDS):
            name_label = QLabel(label)
            value_label = QLabel("Unavailable")
            value_label.setObjectName(f"evaluationMetric_{metric_key}")
            self._metric_labels[metric_key] = value_label
            metrics_layout.addWidget(name_label, row, 0)
            metrics_layout.addWidget(value_label, row, 1)

        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setObjectName("evaluationRunsTable")
        self.runs_table.setHorizontalHeaderLabels(
            [
                "Run Date",
                "Precision@K",
                "Recall@K",
                "MRR",
                "Latency",
                "Duplicate F1",
                "Blur F1",
            ],
        )
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addWidget(title_label)
        layout.addWidget(QLabel("Latest Run Date"))
        layout.addWidget(self.latest_run_date_label)
        layout.addLayout(metrics_layout)
        layout.addWidget(QLabel("Previous Runs"))
        layout.addWidget(self.runs_table, 1)

        self.load_runs()

    def load_runs(self) -> None:
        with self._session_factory_provider()() as session:
            runs = PhotoRepository(session).list_evaluation_runs()

        latest_run = runs[0] if runs else None
        self.latest_run_date_label.setText(
            _format_run_date(latest_run.created_at if latest_run else None),
        )
        self._set_latest_metrics(latest_run)
        self._set_table_runs(runs)

    def _set_latest_metrics(self, latest_run: EvaluationRunData | None) -> None:
        latest_report = latest_run.report if latest_run is not None else {}
        for metric_key, _label in self.METRIC_FIELDS:
            self._metric_labels[metric_key].setText(
                _format_evaluation_metric(latest_report, metric_key),
            )

    def _set_table_runs(self, runs: list[EvaluationRunData]) -> None:
        self.runs_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            values = [
                _format_run_date(run.created_at),
                _format_evaluation_metric(run.report, "precision"),
                _format_evaluation_metric(run.report, "recall"),
                _format_evaluation_metric(run.report, "mean_reciprocal_rank"),
                _format_evaluation_metric(run.report, "average_search_latency_ms"),
                _format_evaluation_metric(run.report, "duplicate_f1"),
                _format_evaluation_metric(run.report, "blur_f1"),
            ]
            for column, value in enumerate(values):
                self.runs_table.setItem(row, column, QTableWidgetItem(value))


class PlaceholderPage(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")

        empty_label = QLabel("Placeholder")
        empty_label.setObjectName("pagePlaceholder")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addWidget(title_label)
        layout.addWidget(empty_label)
        layout.addStretch(1)


class IndexingPage(QWidget):
    def __init__(self) -> None:
        super().__init__()

        title_label = QLabel("Indexing")
        title_label.setObjectName("pageTitle")

        self.source_folder_label = QLabel("No source folder selected")
        self.source_folder_label.setObjectName("selectedSourceFolder")
        self.stage_label = QLabel("Idle")
        self.stage_label.setObjectName("currentStage")
        self.current_filename_label = QLabel("-")
        self.current_filename_label.setObjectName("currentFilename")
        self.counts_label = QLabel("0 / 0")
        self.counts_label.setObjectName("completedTotalCounts")
        self.error_count_label = QLabel("0")
        self.error_count_label.setObjectName("errorCount")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("importProgressBar")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelImportButton")
        self.cancel_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addWidget(title_label)
        layout.addWidget(QLabel("Source Folder"))
        layout.addWidget(self.source_folder_label)
        layout.addWidget(QLabel("Stage"))
        layout.addWidget(self.stage_label)
        layout.addWidget(QLabel("Current Filename"))
        layout.addWidget(self.current_filename_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Completed / Total"))
        layout.addWidget(self.counts_label)
        layout.addWidget(QLabel("Errors"))
        layout.addWidget(self.error_count_label)
        layout.addWidget(self.cancel_button)
        layout.addStretch(1)

    def prepare_for_import(self, folder: Path) -> None:
        self.source_folder_label.setText(str(folder))
        self.stage_label.setText("Starting")
        self.current_filename_label.setText("-")
        self.set_completed_count(0)
        self.set_total_count(0)
        self.set_error_count(0)
        self.progress_bar.setRange(0, 0)
        self.set_running(True)

    def set_stage(self, stage: str) -> None:
        self.stage_label.setText(stage)

    def set_current_filename(self, filename: str) -> None:
        self.current_filename_label.setText(filename or "-")

    def set_completed_count(self, count: int) -> None:
        total = self.progress_bar.maximum()
        self.progress_bar.setValue(count)
        self._set_counts(count, total)

    def set_total_count(self, count: int) -> None:
        if count == 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, count)
        self._set_counts(self.progress_bar.value(), count)

    def set_error_count(self, count: int) -> None:
        self.error_count_label.setText(str(count))

    def set_finished_status(self, status: str) -> None:
        self.set_stage(status)
        self.set_current_filename("")

    def set_running(self, running: bool) -> None:
        self.cancel_button.setEnabled(running)

    def _set_counts(self, completed_count: int, total_count: int) -> None:
        if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            total_count = 0
        self.counts_label.setText(f"{completed_count} / {total_count}")


def _default_database_path() -> Path:
    data_directory = Path.home() / ".lensmind"
    data_directory.mkdir(parents=True, exist_ok=True)
    return data_directory / "lensmind.sqlite3"


def _default_faiss_index_dir() -> Path:
    return Path.home() / ".lensmind" / "faiss"


def _format_run_date(value: datetime | None) -> str:
    if value is None:
        return "Unavailable"
    return value.strftime("%Y-%m-%d %H:%M")


def _format_evaluation_metric(report: dict[str, object], metric_key: str) -> str:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return "Unavailable"

    resolved_key = _resolve_evaluation_metric_key(report, metric_key)
    value = metrics.get(resolved_key)
    if not isinstance(value, int | float):
        return "Unavailable"
    if resolved_key == "average_search_latency_ms":
        return f"{value:.1f} ms"
    return f"{value:.3f}"


def _resolve_evaluation_metric_key(report: dict[str, object], metric_key: str) -> str:
    if metric_key not in {"precision", "recall"}:
        return metric_key
    top_k = report.get("top_k")
    if type(top_k) is int and top_k > 0:
        return f"{metric_key}_at_{top_k}"
    return metric_key


def _format_capture_date(capture_timestamp: datetime | None) -> str:
    if capture_timestamp is None:
        return "Unknown date"
    return capture_timestamp.strftime("%Y-%m-%d")


def _detail_value_label(object_name: str) -> QLabel:
    label = QLabel("-")
    label.setObjectName(object_name)
    label.setWordWrap(True)
    return label


def _format_optional_path(path: Path | None) -> str:
    if path is None:
        return "-"
    return str(path)


def _format_optional_text(value: str | None) -> str:
    if not value:
        return "-"
    return value


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_blur_badge(value: float | None) -> str:
    if value is None:
        return ""
    return f"Blur {value:.0f}"


def _format_similarity_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"Similarity {value:.2f}"


def _trip_page_item_from_trip(
    trip: TripData,
    photo_ids: tuple[int, ...],
) -> TripPageItem:
    return TripPageItem(
        kind="trip",
        name=trip.name,
        photo_ids=photo_ids,
        trip_id=trip.id,
        cover_photo_id=trip.cover_photo_id,
    )


def _trip_page_item_from_suggestion(suggestion: SuggestedTrip) -> TripPageItem:
    return TripPageItem(
        kind="suggestion",
        name=suggestion.name,
        photo_ids=suggestion.photo_ids,
    )


def _format_trip_item(item: TripPageItem) -> str:
    prefix = "Trip" if item.kind == "trip" else "Suggestion"
    return f"{prefix}: {item.name} ({len(item.photo_ids)} photos)"


def _format_trip_status(item: TripPageItem) -> str:
    if item.kind == "trip":
        return f"Saved trip - {len(item.photo_ids)} photos"
    return f"Suggested trip - {len(item.photo_ids)} photos"


def _trip_item_key(item: TripPageItem) -> tuple[str, int | tuple[int, ...]]:
    if item.kind == "trip" and item.trip_id is not None:
        return ("trip", item.trip_id)
    return ("suggestion", item.photo_ids)


def _merged_photo_ids(items: list[TripPageItem]) -> tuple[int, ...]:
    photo_ids: list[int] = []
    seen: set[int] = set()
    for item in items:
        for photo_id in item.photo_ids:
            if photo_id in seen:
                continue
            photo_ids.append(photo_id)
            seen.add(photo_id)
    return tuple(photo_ids)


def _format_duplicate_status(duplicate_group: DuplicateGroupData) -> str:
    if duplicate_group.classification == "exact":
        return "Exact duplicate"
    return "Likely duplicate"


def _format_preferred_photo(duplicate_group: DuplicateGroupData) -> str:
    if duplicate_group.preferred_photo_id is None:
        return "Preferred: none"
    for photo in duplicate_group.photos:
        if photo.id == duplicate_group.preferred_photo_id:
            return f"Preferred: {photo.filename}"
    return "Preferred: unknown"


def _format_duplicate_dimensions(photo: DuplicatePhotoData) -> str:
    if photo.width is None or photo.height is None:
        return "-"
    return _format_dimensions((photo.width, photo.height))


def _format_dimensions(dimensions: tuple[int, int] | None) -> str:
    if dimensions is None:
        return "-"
    width, height = dimensions
    return f"{width} x {height}"


def _format_gps_coordinates(coordinates: tuple[float, float] | None) -> str:
    if coordinates is None:
        return "-"
    latitude, longitude = coordinates
    return f"{latitude:.6f}, {longitude:.6f}"


def _format_file_size(file_size: int) -> str:
    if file_size < 1024:
        return f"{file_size} B"
    if file_size < 1024 * 1024:
        return f"{file_size / 1024:.1f} KB"
    return f"{file_size / (1024 * 1024):.1f} MB"


def run_application(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
