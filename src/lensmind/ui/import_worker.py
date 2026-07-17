from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from lensmind.services.photo_import import PhotoImportProgress, PhotoImportService


class PhotoImportWorker(QObject):
    stage_changed = Signal(str)
    current_filename_changed = Signal(str)
    completed_count_changed = Signal(int)
    total_count_changed = Signal(int)
    error_count_changed = Signal(int)
    finished_status_changed = Signal(str)
    finished = Signal()

    def __init__(self, import_service: PhotoImportService, folder: Path | str) -> None:
        super().__init__()
        self._import_service = import_service
        self._folder = folder
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        status = "failed"
        try:
            summary = self._import_service.import_folder(
                self._folder,
                progress_callback=self._emit_progress,
                should_cancel=self.is_cancelled,
            )
            status = summary.status
        except Exception:
            status = "failed"
        finally:
            self.stage_changed.emit(status)
            self.finished_status_changed.emit(status)
            self.finished.emit()

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _emit_progress(self, progress: PhotoImportProgress) -> None:
        self.stage_changed.emit(progress.stage)
        self.current_filename_changed.emit(progress.current_filename)
        self.completed_count_changed.emit(progress.completed_count)
        self.total_count_changed.emit(progress.total_count)
        self.error_count_changed.emit(progress.error_count)
