from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import PhotoData, PhotoEmbeddingData, PhotoRepository
from lensmind.services.blur_analysis import BlurAnalysisService
from lensmind.services.embeddings import EmbeddingProvider, EmbeddingResult
from lensmind.services.file_hashing import calculate_sha256
from lensmind.services.openclip_embeddings import OpenCLIPEmbeddingProvider
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


@dataclass(frozen=True)
class PendingEmbedding:
    photo_id: int
    path: Path
    filename: str


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
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._metadata_extractor = metadata_extractor or PhotoMetadataExtractor()
        self._blur_analysis_service = blur_analysis_service or BlurAnalysisService()
        self._thumbnail_generator = thumbnail_generator or ThumbnailGenerator(
            _default_thumbnail_cache_dir(),
        )
        self._embedding_provider = embedding_provider or OpenCLIPEmbeddingProvider()

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
            pending_embeddings: list[PendingEmbedding] = []
            files_imported = 0
            files_added = 0
            status = "completed"
            embedding_model_name = _embedding_model_name(self._embedding_provider)
            embedding_model_config = _embedding_model_config(
                self._embedding_provider,
            )

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

                photo = repository.add_or_update_photo(photo_data)
                if repository.photo_needs_embedding(
                    photo.id,
                    model_name=embedding_model_name,
                    model_config=embedding_model_config,
                ):
                    pending_embeddings.append(
                        PendingEmbedding(
                            photo_id=photo.id,
                            path=record.path,
                            filename=record.filename,
                        ),
                    )
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

            if status != "cancelled":
                status = self._generate_embeddings(
                    repository,
                    pending_embeddings,
                    errors,
                    progress_callback,
                    should_cancel,
                    status,
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

    def _generate_embeddings(
        self,
        repository: PhotoRepository,
        pending_embeddings: list[PendingEmbedding],
        errors: list[PhotoImportError],
        progress_callback: ProgressCallback | None,
        should_cancel: CancellationCallback | None,
        status: str,
    ) -> str:
        total_count = len(pending_embeddings)
        completed_count = 0
        _emit_progress(progress_callback, "embedding", "", 0, total_count, len(errors))

        for batch in _batches(
            pending_embeddings,
            _embedding_batch_size(self._embedding_provider),
        ):
            if should_cancel is not None and should_cancel():
                return "cancelled"

            current_filename = batch[0].filename if batch else ""
            _emit_progress(
                progress_callback,
                "embedding",
                current_filename,
                completed_count,
                total_count,
                len(errors),
            )
            try:
                results = self._embedding_provider.embed_images(
                    [pending.path for pending in batch],
                )
            except Exception as error:
                for pending in batch:
                    errors.append(PhotoImportError(pending.path, str(error)))
                    completed_count += 1
                    _emit_progress(
                        progress_callback,
                        "embedding",
                        pending.filename,
                        completed_count,
                        total_count,
                        len(errors),
                    )
                continue

            for pending, result in zip(batch, results, strict=False):
                self._save_embedding_result(repository, pending, result, errors)
                completed_count += 1
                _emit_progress(
                    progress_callback,
                    "embedding",
                    pending.filename,
                    completed_count,
                    total_count,
                    len(errors),
                )

        return status

    def _save_embedding_result(
        self,
        repository: PhotoRepository,
        pending: PendingEmbedding,
        result: EmbeddingResult,
        errors: list[PhotoImportError],
    ) -> None:
        if result.error is not None:
            errors.append(PhotoImportError(pending.path, result.error))
            return
        if result.vector is None:
            errors.append(PhotoImportError(pending.path, "missing embedding vector"))
            return

        repository.save_photo_embedding(
            PhotoEmbeddingData(
                photo_id=pending.photo_id,
                model_name=result.model_name,
                model_config=result.pretrained,
                vector_dimension=result.dimension,
                embedding_data=_serialize_embedding(result.vector),
            ),
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


def _serialize_embedding(vector: tuple[float, ...]) -> bytes:
    return array("f", vector).tobytes()


def _embedding_model_name(provider: EmbeddingProvider) -> str:
    return str(getattr(provider, "model_name", "unknown"))


def _embedding_model_config(provider: EmbeddingProvider) -> str:
    return str(getattr(provider, "pretrained", "unknown"))


def _embedding_batch_size(provider: EmbeddingProvider) -> int:
    try:
        batch_size = int(getattr(provider, "batch_size", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, batch_size)


def _batches(
    items: list[PendingEmbedding],
    batch_size: int,
) -> list[list[PendingEmbedding]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _default_thumbnail_cache_dir() -> Path:
    return Path.home() / ".lensmind" / "thumbnails"
