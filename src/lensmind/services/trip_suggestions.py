from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

from sqlalchemy.orm import Session

from lensmind.db.models import Photo
from lensmind.db.repository import PhotoRepository

DEFAULT_MAX_TIME_GAP = timedelta(hours=36)
DEFAULT_MAX_DISTANCE_KM = 250.0
MIN_SUGGESTED_TRIP_PHOTOS = 2
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class SuggestedTrip:
    name: str
    photo_ids: tuple[int, ...]
    start_date: datetime
    end_date: datetime


class TripSuggestionService:
    def __init__(self, session: Session) -> None:
        self._repository = PhotoRepository(session)

    def suggest_trips(
        self,
        *,
        max_time_gap: timedelta = DEFAULT_MAX_TIME_GAP,
        max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    ) -> list[SuggestedTrip]:
        photos = [
            photo
            for photo in self._repository.list_photos()
            if photo.capture_timestamp is not None
        ]
        photos.sort(key=lambda photo: (photo.capture_timestamp, photo.id))
        if not photos:
            return []

        groups: list[list[Photo]] = []
        current_group = [photos[0]]
        for photo in photos[1:]:
            previous = current_group[-1]
            if _starts_new_group(
                previous,
                photo,
                max_time_gap=max_time_gap,
                max_distance_km=max_distance_km,
            ):
                groups.append(current_group)
                current_group = [photo]
            else:
                current_group.append(photo)
        groups.append(current_group)

        return [
            _suggested_trip(group)
            for group in groups
            if len(group) >= MIN_SUGGESTED_TRIP_PHOTOS
        ]


def _starts_new_group(
    previous: Photo,
    photo: Photo,
    *,
    max_time_gap: timedelta,
    max_distance_km: float,
) -> bool:
    if previous.capture_timestamp is None or photo.capture_timestamp is None:
        return True

    if photo.capture_timestamp - previous.capture_timestamp > max_time_gap:
        return True

    distance_km = _photo_distance_km(previous, photo)
    return distance_km is not None and distance_km > max_distance_km


def _photo_distance_km(first: Photo, second: Photo) -> float | None:
    if (
        first.latitude is None
        or first.longitude is None
        or second.latitude is None
        or second.longitude is None
    ):
        return None

    return _haversine_km(
        first.latitude,
        first.longitude,
        second.latitude,
        second.longitude,
    )


def _haversine_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    first_latitude_rad = radians(first_latitude)
    second_latitude_rad = radians(second_latitude)
    latitude_delta = radians(second_latitude - first_latitude)
    longitude_delta = radians(second_longitude - first_longitude)
    half_chord = (
        sin(latitude_delta / 2) ** 2
        + cos(first_latitude_rad)
        * cos(second_latitude_rad)
        * sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * asin(sqrt(half_chord))


def _suggested_trip(photos: list[Photo]) -> SuggestedTrip:
    start_date = photos[0].capture_timestamp
    end_date = photos[-1].capture_timestamp
    if start_date is None or end_date is None:
        msg = "suggested trip photos must have capture timestamps"
        raise ValueError(msg)

    return SuggestedTrip(
        name=_date_based_name(start_date, end_date),
        photo_ids=tuple(photo.id for photo in photos),
        start_date=start_date,
        end_date=end_date,
    )


def _date_based_name(start_date: datetime, end_date: datetime) -> str:
    start_text = start_date.strftime("%Y-%m-%d")
    end_text = end_date.strftime("%Y-%m-%d")
    if start_text == end_text:
        return f"Trip {start_text}"
    return f"Trip {start_text} to {end_text}"
