from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from lensmind.db.repository import PhotoData, PhotoRepository, initialize_sqlite
from lensmind.services.trip_suggestions import TripSuggestionService


def test_suggests_date_named_groups_from_capture_time_gaps(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        first = add_photo(repository, "first.jpg", datetime(2026, 1, 1, 9))
        second = add_photo(repository, "second.jpg", datetime(2026, 1, 1, 12))
        third = add_photo(repository, "third.jpg", datetime(2026, 1, 5, 9))
        fourth = add_photo(repository, "fourth.jpg", datetime(2026, 1, 5, 11))

        suggestions = TripSuggestionService(session).suggest_trips(
            max_time_gap=timedelta(hours=24),
        )

    assert [(trip.name, trip.photo_ids) for trip in suggestions] == [
        ("Trip 2026-01-01", (first.id, second.id)),
        ("Trip 2026-01-05", (third.id, fourth.id)),
    ]


def test_uses_gps_distance_when_adjacent_photos_have_coordinates(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        toronto = add_photo(
            repository,
            "toronto.jpg",
            datetime(2026, 2, 1, 9),
            latitude=43.6532,
            longitude=-79.3832,
        )
        nearby = add_photo(
            repository,
            "nearby.jpg",
            datetime(2026, 2, 1, 10),
            latitude=43.66,
            longitude=-79.39,
        )
        london = add_photo(
            repository,
            "london.jpg",
            datetime(2026, 2, 1, 11),
            latitude=51.5072,
            longitude=-0.1276,
        )
        paris = add_photo(
            repository,
            "paris.jpg",
            datetime(2026, 2, 1, 12),
            latitude=48.8566,
            longitude=2.3522,
        )

        suggestions = TripSuggestionService(session).suggest_trips(
            max_time_gap=timedelta(hours=24),
            max_distance_km=400.0,
        )

    assert [(trip.name, trip.photo_ids) for trip in suggestions] == [
        ("Trip 2026-02-01", (toronto.id, nearby.id)),
        ("Trip 2026-02-01", (london.id, paris.id)),
    ]


def test_ignores_photos_without_capture_dates_and_singletons(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        add_photo(repository, "missing-date.jpg", None)
        add_photo(repository, "single.jpg", datetime(2026, 3, 1, 9))

        suggestions = TripSuggestionService(session).suggest_trips()

    assert suggestions == []


def test_uses_date_range_name_for_multi_day_groups(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        first = add_photo(repository, "first.jpg", datetime(2026, 4, 1, 22))
        second = add_photo(repository, "second.jpg", datetime(2026, 4, 2, 8))

        suggestions = TripSuggestionService(session).suggest_trips(
            max_time_gap=timedelta(hours=24),
        )

    assert [(trip.name, trip.photo_ids) for trip in suggestions] == [
        ("Trip 2026-04-01 to 2026-04-02", (first.id, second.id)),
    ]


def add_photo(
    repository: PhotoRepository,
    filename: str,
    capture_timestamp: datetime | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
):
    return repository.add_or_update_photo(
        PhotoData(
            original_path=f"/photos/{filename}",
            filename=filename,
            file_size=100,
            capture_timestamp=capture_timestamp,
            latitude=latitude,
            longitude=longitude,
        ),
    )
