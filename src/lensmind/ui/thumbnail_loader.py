from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage

ThumbnailStatus = Literal["loaded", "missing", "error"]


@dataclass(frozen=True)
class ThumbnailLoadResult:
    path: Path
    status: ThumbnailStatus
    image: QImage | None = None
    error: str | None = None


class ThumbnailLoadSignals(QObject):
    finished = Signal(object)


class ThumbnailLoadTask(QRunnable):
    def __init__(self, path: Path | str, size: QSize) -> None:
        super().__init__()
        self.signals = ThumbnailLoadSignals()
        self._path = Path(path)
        self._size = size

    @Slot()
    def run(self) -> None:
        if not self._path.exists():
            self.signals.finished.emit(
                ThumbnailLoadResult(path=self._path, status="missing"),
            )
            return

        image = QImage(str(self._path))
        if image.isNull():
            self.signals.finished.emit(
                ThumbnailLoadResult(
                    path=self._path,
                    status="error",
                    error="Unable to load thumbnail",
                ),
            )
            return

        self.signals.finished.emit(
            ThumbnailLoadResult(
                path=self._path,
                status="loaded",
                image=image.scaled(
                    self._size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ),
            ),
        )


class ThumbnailLoader:
    def __init__(self, thread_pool: QThreadPool | None = None) -> None:
        self._thread_pool = thread_pool or QThreadPool.globalInstance()

    def load(
        self,
        path: Path | str,
        size: QSize,
        callback: Callable[[ThumbnailLoadResult], None],
    ) -> ThumbnailLoadTask:
        task = ThumbnailLoadTask(path, size)
        task.signals.finished.connect(callback)
        self._thread_pool.start(task)
        return task
