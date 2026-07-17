from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.models import Photo
from lensmind.db.repository import PhotoRepository, initialize_sqlite
from lensmind.services.photo_import import PhotoImportService
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


class MainWindow(QMainWindow):
    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        super().__init__()

        self.setWindowTitle("LensMind")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1280, 800)

        self._session_factory = session_factory
        self._import_thread: QThread | None = None
        self._import_worker: PhotoImportWorker | None = None
        self._all_photos_page = AllPhotosPage(self._get_session_factory)
        self._indexing_page = IndexingPage()
        self._pages = QStackedWidget()
        self._sidebar = self._build_sidebar()
        self._inspector = self._build_inspector()

        self._build_toolbar()
        self._build_pages()
        self._build_layout()

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

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_sidebar(self) -> QListWidget:
        sidebar = QListWidget()
        sidebar.setFixedWidth(220)
        sidebar.setSpacing(2)
        sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sidebar.currentRowChanged.connect(self._pages.setCurrentIndex)
        sidebar.currentRowChanged.connect(self._handle_page_changed)

        for title in PAGE_TITLES:
            item = QListWidgetItem(title)
            item.setSizeHint(QSize(180, 36))
            sidebar.addItem(item)

        return sidebar

    def _build_pages(self) -> None:
        for title in PAGE_TITLES:
            if title == "All Photos":
                self._pages.addWidget(self._all_photos_page)
            elif title == "Indexing":
                self._pages.addWidget(self._indexing_page)
            else:
                self._pages.addWidget(PlaceholderPage(title))

    def _build_inspector(self) -> QFrame:
        inspector = QFrame()
        inspector.setObjectName("rightInspector")
        inspector.setMinimumWidth(260)
        inspector.setMaximumWidth(340)
        inspector.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(QLabel("Inspector"))
        layout.addStretch(1)
        return inspector

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
        self._indexing_page.cancel_button.clicked.connect(worker.cancel)

        thread.start()

    def _handle_page_changed(self, row: int) -> None:
        if PAGE_TITLES[row] == "All Photos":
            self._all_photos_page.load_photos()

    def _clear_import_worker(self) -> None:
        self._indexing_page.cancel_button.clicked.disconnect()
        self._indexing_page.set_running(False)
        self._import_folder_action.setEnabled(True)
        self._import_worker = None
        self._import_thread = None

    def _get_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = initialize_sqlite(_default_database_path())

        return self._session_factory


class AllPhotosPage(QWidget):
    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]],
    ) -> None:
        super().__init__()
        self._session_factory_provider = session_factory_provider
        self._thumbnail_loader = ThumbnailLoader()

        title_label = QLabel("All Photos")
        title_label.setObjectName("pageTitle")

        self._empty_label = QLabel("No photos imported yet")
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
        session_factory = self._session_factory_provider()
        with session_factory() as session:
            photos = PhotoRepository(session).list_photos()

        self._empty_label.setVisible(not photos)
        for index, photo in enumerate(photos):
            row, column = divmod(index, 4)
            self._grid.addWidget(
                PhotoGridItem(photo, self._thumbnail_loader),
                row,
                column,
            )

        self._grid.setRowStretch((len(photos) // 4) + 1, 1)

    def _clear_grid(self) -> None:
        while item := self._grid.takeAt(0):
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class PhotoGridItem(QFrame):
    def __init__(self, photo: Photo, thumbnail_loader: ThumbnailLoader) -> None:
        super().__init__()
        self.setObjectName("photoGridItem")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(180)
        self._thumbnail_path = (
            Path(photo.thumbnail_path) if photo.thumbnail_path else None
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.thumbnail_label)
        layout.addWidget(self.thumbnail_state_label)
        layout.addWidget(filename_label)
        layout.addWidget(capture_date_label)

        self._load_thumbnail(photo, thumbnail_loader)

    def _load_thumbnail(
        self,
        photo: Photo,
        thumbnail_loader: ThumbnailLoader,
    ) -> None:
        if photo.missing_file:
            self._set_thumbnail_state("Missing file", clear_image=True)
            return

        if self._thumbnail_path is None:
            self._set_thumbnail_state("No thumbnail", clear_image=True)
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


def _format_capture_date(capture_timestamp: datetime | None) -> str:
    if capture_timestamp is None:
        return "Unknown date"
    return capture_timestamp.strftime("%Y-%m-%d")


def run_application(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
