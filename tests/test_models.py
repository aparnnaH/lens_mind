from sqlalchemy import create_engine, inspect

from lensmind.db import Base
from lensmind.db.models import Photo


def test_models_create_expected_tables() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    table_names = set(inspect(engine).get_table_names())
    assert table_names == {
        "album_photos",
        "albums",
        "indexing_runs",
        "photos",
        "source_folders",
    }


def test_photo_model_has_requested_columns() -> None:
    expected_columns = {
        "id",
        "original_path",
        "source_folder_id",
        "filename",
        "file_size",
        "sha256",
        "capture_timestamp",
        "timestamp_source",
        "width",
        "height",
        "camera_make",
        "camera_model",
        "latitude",
        "longitude",
        "blur_score",
        "thumbnail_path",
        "processing_status",
        "processing_error",
        "missing_file",
    }

    assert set(Photo.__table__.columns.keys()) == expected_columns
