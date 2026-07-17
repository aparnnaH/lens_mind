from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lensmind.db.repository import PhotoData, PhotoRepository, initialize_sqlite
from lensmind.services.photo_display import PhotoDisplayService


def test_photo_display_service_returns_complete_photo_info(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    capture_date = datetime(2026, 4, 5, 6, 7, 8)

    with session_factory() as session:
        repository = PhotoRepository(session)
        source_folder = repository.add_source_folder(str(tmp_path / "photos"))
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "photos" / "image.jpg"),
                source_folder_id=source_folder.id,
                filename="image.jpg",
                file_size=2048,
                capture_timestamp=capture_date,
                timestamp_source="exif",
                width=4000,
                height=3000,
                camera_make="Canon",
                camera_model="EOS",
                latitude=43.65,
                longitude=-79.38,
                blur_score=0.12,
                thumbnail_path=str(tmp_path / "thumbs" / "image.png"),
            ),
        )

        display_info = PhotoDisplayService(session).get_photo_display_info(photo.id)

    assert display_info is not None
    assert display_info.preview_path == tmp_path / "thumbs" / "image.png"
    assert display_info.filename == "image.jpg"
    assert display_info.original_path == tmp_path / "photos" / "image.jpg"
    assert display_info.file_size == 2048
    assert display_info.capture_date == capture_date
    assert display_info.timestamp_source == "exif"
    assert display_info.dimensions == (4000, 3000)
    assert display_info.camera_details == "Canon EOS"
    assert display_info.gps_coordinates == (43.65, -79.38)
    assert display_info.blur_score == 0.12
    assert display_info.source_folder == tmp_path / "photos"
    assert display_info.missing_file is False


def test_photo_display_service_handles_missing_photo_and_missing_file(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path=str(tmp_path / "missing.jpg"),
                filename="missing.jpg",
                file_size=100,
                missing_file=True,
            ),
        )
        service = PhotoDisplayService(session)
        display_info = service.get_photo_display_info(photo.id)

        assert service.get_photo_display_info(999) is None

    assert display_info is not None
    assert display_info.preview_path is None
    assert display_info.source_folder is None
    assert display_info.missing_file is True
