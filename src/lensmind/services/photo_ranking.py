from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from lensmind.db.models import Photo
from lensmind.db.repository import PhotoRepository

MAX_BLUR_COMPONENT = 40.0
MAX_RESOLUTION_COMPONENT = 30.0
DUPLICATE_PREFERRED_BONUS = 20.0
DUPLICATE_NOT_PREFERRED_PENALTY = 15.0
MAX_EXPOSURE_CLIPPING_PENALTY = 20.0
REFERENCE_BLUR_SCORE = 500.0
REFERENCE_MEGAPIXELS = 12.0


@dataclass(frozen=True)
class ExposureClipping:
    shadow_fraction: float = 0.0
    highlight_fraction: float = 0.0


@dataclass(frozen=True)
class PhotoRank:
    photo_id: int
    score: float
    reasons: tuple[str, ...]


class PhotoRankingService:
    def __init__(
        self,
        session: Session,
        *,
        exposure_clipping_by_photo_id: dict[int, ExposureClipping] | None = None,
    ) -> None:
        self._repository = PhotoRepository(session)
        self._exposure_clipping_by_photo_id = exposure_clipping_by_photo_id or {}

    def rank_photos(self, photos: list[Photo] | None = None) -> list[PhotoRank]:
        photo_list = photos if photos is not None else self._repository.list_photos()
        duplicate_preferences = self._duplicate_preferences()
        ranks = [
            self.rank_photo(
                photo,
                duplicate_preference=duplicate_preferences.get(photo.id),
            )
            for photo in photo_list
        ]
        return sorted(ranks, key=lambda rank: (-rank.score, rank.photo_id))

    def rank_photo(
        self,
        photo: Photo,
        *,
        duplicate_preference: bool | None = None,
    ) -> PhotoRank:
        score = 0.0
        reasons: list[str] = []

        blur_score, blur_reason = _blur_component(photo.blur_score)
        score += blur_score
        reasons.append(blur_reason)

        resolution_score, resolution_reason = _resolution_component(
            photo.width,
            photo.height,
        )
        score += resolution_score
        reasons.append(resolution_reason)

        duplicate_score, duplicate_reason = _duplicate_component(
            duplicate_preference,
        )
        score += duplicate_score
        reasons.append(duplicate_reason)

        clipping = self._exposure_clipping_by_photo_id.get(photo.id)
        exposure_penalty, exposure_reason = _exposure_component(clipping)
        score -= exposure_penalty
        reasons.append(exposure_reason)

        return PhotoRank(
            photo_id=photo.id,
            score=round(max(0.0, min(100.0, score)), 2),
            reasons=tuple(reasons),
        )

    def _duplicate_preferences(self) -> dict[int, bool]:
        preferences: dict[int, bool] = {}
        for group in self._repository.list_duplicate_groups():
            if group.preferred_photo_id is None:
                continue
            for photo in group.photos:
                preferences[photo.id] = photo.id == group.preferred_photo_id
        return preferences


def _blur_component(blur_score: float | None) -> tuple[float, str]:
    if blur_score is None:
        return MAX_BLUR_COMPONENT / 2, "No blur score"

    component = min(
        MAX_BLUR_COMPONENT,
        max(0.0, (blur_score / REFERENCE_BLUR_SCORE) * MAX_BLUR_COMPONENT),
    )
    if component < MAX_BLUR_COMPONENT / 2:
        return component, "Low blur score"
    return component, "Higher blur score"


def _resolution_component(
    width: int | None,
    height: int | None,
) -> tuple[float, str]:
    if width is None or height is None:
        return MAX_RESOLUTION_COMPONENT / 2, "No resolution"

    megapixels = (width * height) / 1_000_000
    component = min(
        MAX_RESOLUTION_COMPONENT,
        max(0.0, (megapixels / REFERENCE_MEGAPIXELS) * MAX_RESOLUTION_COMPONENT),
    )
    if megapixels < 2:
        return component, "Low resolution"
    return component, "Higher resolution"


def _duplicate_component(preferred: bool | None) -> tuple[float, str]:
    if preferred is True:
        return DUPLICATE_PREFERRED_BONUS, "Preferred duplicate"
    if preferred is False:
        return -DUPLICATE_NOT_PREFERRED_PENALTY, "Not preferred duplicate"
    return 0.0, "No duplicate preference"


def _exposure_component(
    clipping: ExposureClipping | None,
) -> tuple[float, str]:
    if clipping is None:
        return 0.0, "No exposure clipping data"

    clipped_fraction = max(
        0.0,
        min(1.0, clipping.shadow_fraction + clipping.highlight_fraction),
    )
    penalty = clipped_fraction * MAX_EXPOSURE_CLIPPING_PENALTY
    if penalty == 0:
        return 0.0, "No exposure clipping"
    return penalty, "Exposure clipping"
