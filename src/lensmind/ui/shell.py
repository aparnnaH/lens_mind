from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

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
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("LensMind")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1280, 800)

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


def run_application(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
