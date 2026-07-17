from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import PhotoData, PhotoRepository
from lensmind.services.blur_analysis import BlurAnalysisService
from lensmind.services.file_hashing import calculate_sha256
from lensmind.services.photo_discovery import PhotoFileRecord, discover_photo_files
from lensmind.services.photo_metadata import PhotoMetadataExtractor
from lensmind.services.thumbnail_generation import ThumbnailGenerator


@dataclass(frozen=True)
class PhotoImportError:
    path: Path
    message: str


@dataclass(frozen=True)
class PhotoImportSummary:
    source_folder_id: int
    indexing_run_id: int
    files_seen: int
    files_imported: int
    files_added: int
    errors: tuple[PhotoImportError, ...]
    status: str


@dataclass(frozen=True)
class PhotoImportProgress:
    stage: str
    current_filename: str
    completed_count: int
    total_count: int
    error_count: int


class ProgressCallback(Protocol):
    def __call__(self, progress: PhotoImportProgress) -> None: ...


class CancellationCallback(Protocol):
    def __call__(self) -> bool: ...


class PhotoImportService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        metadata_extractor: PhotoMetadataExtractor | None = None,
        blur_analysis_service: BlurAnalysisService | None = None,
        thumbnail_generator: ThumbnailGenerator | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._metadata_extractor = metadata_extractor or PhotoMetadataExtractor()
        self._blur_analysis_service = blur_analysis_service or BlurAnalysisService()
        self._thumbnail_generator = thumbnail_generator or ThumbnailGenerator(
            _default_thumbnail_cache_dir(),
        )

    def import_folder(
        self,
        folder: Path | str,
        progress_callback: ProgressCallback | None = None,
        should_cancel: CancellationCallback | None = None,
    ) -> PhotoImportSummary:
        source_path = Path(folder).expanduser()
        started_at = datetime.now(UTC)
        _emit_progress(progress_callback, "discovering", "", 0, 0, 0)
        records = discover_photo_files(source_path)
        errors: list[PhotoImportError] = []
        total_count = len(records)
        completed_count = 0
        _emit_progress(progress_callback, "importing", "", 0, total_count, 0)

        with self._session_factory() as session:
            repository = PhotoRepository(session)
            source_folder = repository.add_source_folder(str(source_path))
            existing_paths = {
                Path(photo.original_path)
                for photo in repository.list_photos()
            }
            files_imported = 0
            files_added = 0
            status = "completed"

            for record in records:
                if should_cancel is not None and should_cancel():
                    status = "cancelled"
                    break

                _emit_progress(
                    progress_callback,
                    "importing",
                    record.filename,
                    completed_count,
                    total_count,
                    len(errors),
                )
                photo_data = self._build_photo_data(record, errors, source_folder.id)
                completed_count += 1
                if photo_data is None:
                    _emit_progress(
                        progress_callback,
                        "importing",
                        record.filename,
                        completed_count,
                        total_count,
                        len(errors),
                    )
                    continue

                repository.add_or_update_photo(photo_data)
                files_imported += 1
                if record.path not in existing_paths:
                    files_added += 1

                _emit_progress(
                    progress_callback,
                    "importing",
                    record.filename,
                    completed_count,
                    total_count,
                    len(errors),
                )

            if status != "cancelled" and errors:
                status = "completed_with_errors"

            _emit_progress(
                progress_callback,
                "recording",
                "",
                completed_count,
                total_count,
                len(errors),
            )
            indexing_run = repository.record_indexing_run(
                source_folder.id,
                status=status,
                files_seen=len(records),
                files_added=files_added,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                error="\n".join(error.message for error in errors) or None,
            )

            return PhotoImportSummary(
                source_folder_id=source_folder.id,
                indexing_run_id=indexing_run.id,
                files_seen=len(records),
                files_imported=files_imported,
                files_added=files_added,
                errors=tuple(errors),
                status=status,
            )

    def _build_photo_data(
        self,
        record: PhotoFileRecord,
        errors: list[PhotoImportError],
        source_folder_id: int,
    ) -> PhotoData | None:
        try:
            sha256 = calculate_sha256(record.path)
        except OSError as error:
            errors.append(PhotoImportError(record.path, str(error)))
            return None

        metadata = self._metadata_extractor.extract(record.path)
        if metadata.error is not None:
            errors.append(PhotoImportError(record.path, metadata.error))

        blur_result = self._blur_analysis_service.analyze(record.path)
        if blur_result.error is not None and metadata.error is None:
            errors.append(PhotoImportError(record.path, blur_result.error))

        thumbnail_result = self._thumbnail_generator.generate(record.path)
        if (
            thumbnail_result.error is not None
            and metadata.error is None
            and blur_result.error is None
        ):
            errors.append(PhotoImportError(record.path, thumbnail_result.error))

        processing_error = metadata.error or blur_result.error or thumbnail_result.error

        return PhotoData(
            original_path=str(record.path),
            filename=record.filename,
            file_size=record.file_size,
            source_folder_id=source_folder_id,
            sha256=sha256,
            capture_timestamp=metadata.capture_timestamp,
            timestamp_source=metadata.timestamp_source,
            width=metadata.width,
            height=metadata.height,
            camera_make=metadata.camera_make,
            camera_model=metadata.camera_model,
            latitude=metadata.latitude,
            longitude=metadata.longitude,
            blur_score=blur_result.raw_score,
            thumbnail_path=(
                str(thumbnail_result.thumbnail_path)
                if thumbnail_result.thumbnail_path is not None
                else None
            ),
            processing_status=_processing_status(
                metadata.error,
                blur_result.error,
                thumbnail_result.error,
            ),
            processing_error=processing_error,
            missing_file=False,
        )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    current_filename: str,
    completed_count: int,
    total_count: int,
    error_count: int,
) -> None:
    if progress_callback is None:
        return

    progress_callback(
        PhotoImportProgress(
            stage=stage,
            current_filename=current_filename,
            completed_count=completed_count,
            total_count=total_count,
            error_count=error_count,
        ),
    )


def _processing_status(
    metadata_error: str | None,
    blur_error: str | None,
    thumbnail_error: str | None,
) -> str:
    if metadata_error is not None:
        return "metadata_error"
    if blur_error is not None:
        return "analysis_error"
    if thumbnail_error is not None:
        return "thumbnail_error"
    return "imported"


def _default_thumbnail_cache_dir() -> Path:
    return Path.home() / ".lensmind" / "thumbnails"
