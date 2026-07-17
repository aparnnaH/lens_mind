from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import initialize_sqlite
from lensmind.services.photo_import import PhotoImportService
from lensmind.ui.import_worker import PhotoImportWorker

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

        for title in PAGE_TITLES:
            item = QListWidgetItem(title)
            item.setSizeHint(QSize(180, 36))
            sidebar.addItem(item)

        return sidebar

    def _build_pages(self) -> None:
        for title in PAGE_TITLES:
            if title == "Indexing":
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
        self._indexing_page.cancel_button.clicked.connect(worker.cancel)

        thread.start()

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


def run_application(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
