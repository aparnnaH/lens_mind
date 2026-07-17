from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import PhotoData, PhotoRepository
from lensmind.services.file_hashing import calculate_sha256
from lensmind.services.photo_discovery import PhotoFileRecord, discover_photo_files
from lensmind.services.photo_metadata import PhotoMetadataExtractor


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


class PhotoImportService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        metadata_extractor: PhotoMetadataExtractor | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._metadata_extractor = metadata_extractor or PhotoMetadataExtractor()

    def import_folder(self, folder: Path | str) -> PhotoImportSummary:
        source_path = Path(folder).expanduser()
        started_at = datetime.now(UTC)
        records = discover_photo_files(source_path)
        errors: list[PhotoImportError] = []

        with self._session_factory() as session:
            repository = PhotoRepository(session)
            source_folder = repository.add_source_folder(str(source_path))
            existing_paths = {
                Path(photo.original_path)
                for photo in repository.list_photos()
            }
            files_imported = 0
            files_added = 0

            for record in records:
                photo_data = self._build_photo_data(record, errors)
                if photo_data is None:
                    continue

                repository.add_or_update_photo(photo_data)
                files_imported += 1
                if record.path not in existing_paths:
                    files_added += 1

            status = "completed_with_errors" if errors else "completed"
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
            )

    def _build_photo_data(
        self,
        record: PhotoFileRecord,
        errors: list[PhotoImportError],
    ) -> PhotoData | None:
        try:
            sha256 = calculate_sha256(record.path)
        except OSError as error:
            errors.append(PhotoImportError(record.path, str(error)))
            return None

        metadata = self._metadata_extractor.extract(record.path)
        if metadata.error is not None:
            errors.append(PhotoImportError(record.path, metadata.error))

        return PhotoData(
            original_path=str(record.path),
            filename=record.filename,
            file_size=record.file_size,
            sha256=sha256,
            capture_timestamp=metadata.capture_timestamp,
            width=metadata.width,
            height=metadata.height,
            camera_make=metadata.camera_make,
            camera_model=metadata.camera_model,
            latitude=metadata.latitude,
            longitude=metadata.longitude,
            processing_status="metadata_error" if metadata.error else "imported",
            processing_error=metadata.error,
            missing_file=False,
        )
