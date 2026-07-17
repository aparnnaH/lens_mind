from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from lensmind.ui import shell  # noqa: E402


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


def test_main_window_creates_navigation_pages(app: QApplication) -> None:
    window = shell.MainWindow()
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
