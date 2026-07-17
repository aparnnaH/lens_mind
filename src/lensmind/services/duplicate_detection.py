from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from lensmind.db.models import DuplicateGroup, DuplicateGroupPhoto, Photo
from lensmind.services.perceptual_hashing import (
    PerceptualHashService,
    hamming_distance,
)


@dataclass(frozen=True)
class DuplicateDetectionSummary:
    groups_created: int
    duplicate_photos: int
    exact_groups_created: int = 0
    likely_groups_created: int = 0


class DuplicateDetectionService:
    def __init__(
        self,
        session: Session,
        perceptual_hash_service: PerceptualHashService | None = None,
        distance_threshold: int = 8,
    ) -> None:
        self._session = session
        self._perceptual_hash_service = (
            perceptual_hash_service or PerceptualHashService()
        )
        self._distance_threshold = distance_threshold

    def rebuild_duplicate_groups(self) -> DuplicateDetectionSummary:
        self._clear_existing_groups()
        duplicate_hashes = self._duplicate_hashes()
        exact_groups_created = 0
        likely_groups_created = 0
        duplicate_photo_count = 0

        for sha256 in duplicate_hashes:
            photos = list(
                self._session.scalars(
                    select(Photo).where(Photo.sha256 == sha256).order_by(Photo.id),
                ),
            )
            duplicate_photo_count += len(photos)
            self._create_group(
                classification="exact",
                group_key=f"exact:{sha256}",
                photos=photos,
                sha256=sha256,
            )
            exact_groups_created += 1

        self._ensure_perceptual_hashes()
        for first_photo, second_photo, distance in self._likely_duplicate_pairs():
            self._create_group(
                classification="likely",
                group_key=f"likely:{first_photo.id}:{second_photo.id}",
                photos=[first_photo, second_photo],
                distance_threshold=self._distance_threshold,
                max_distance=distance,
            )
            likely_groups_created += 1
            duplicate_photo_count += 2

        self._session.commit()
        return DuplicateDetectionSummary(
            groups_created=exact_groups_created + likely_groups_created,
            duplicate_photos=duplicate_photo_count,
            exact_groups_created=exact_groups_created,
            likely_groups_created=likely_groups_created,
        )

    def _clear_existing_groups(self) -> None:
        self._session.execute(delete(DuplicateGroupPhoto))
        self._session.execute(delete(DuplicateGroup))

    def _duplicate_hashes(self) -> list[str]:
        return list(
            self._session.scalars(
                select(Photo.sha256)
                .where(Photo.sha256.is_not(None))
                .where(Photo.sha256 != "")
                .group_by(Photo.sha256)
                .having(func.count(Photo.id) > 1)
                .order_by(Photo.sha256),
            ),
        )

    def _ensure_perceptual_hashes(self) -> None:
        photos = self._session.scalars(
            select(Photo)
            .where(Photo.perceptual_hash.is_(None))
            .where(Photo.missing_file.is_(False))
            .order_by(Photo.id),
        )
        for photo in photos:
            result = self._perceptual_hash_service.compute(photo.original_path)
            if result.value is not None:
                photo.perceptual_hash = result.value
        self._session.flush()

    def _likely_duplicate_pairs(self) -> list[tuple[Photo, Photo, int]]:
        photos = list(
            self._session.scalars(
                select(Photo)
                .where(Photo.perceptual_hash.is_not(None))
                .order_by(Photo.id),
            ),
        )
        likely_pairs: list[tuple[Photo, Photo, int]] = []
        for first_photo, second_photo in combinations(photos, 2):
            if _same_exact_hash(first_photo, second_photo):
                continue
            if (
                first_photo.perceptual_hash is None
                or second_photo.perceptual_hash is None
            ):
                continue
            distance = hamming_distance(
                first_photo.perceptual_hash,
                second_photo.perceptual_hash,
            )
            if distance <= self._distance_threshold:
                likely_pairs.append((first_photo, second_photo, distance))
        return likely_pairs

    def _create_group(
        self,
        *,
        classification: str,
        group_key: str,
        photos: list[Photo],
        sha256: str | None = None,
        distance_threshold: int | None = None,
        max_distance: int | None = None,
    ) -> None:
        duplicate_group = DuplicateGroup(
            classification=classification,
            group_key=group_key,
            sha256=sha256,
            distance_threshold=distance_threshold,
            max_distance=max_distance,
        )
        self._session.add(duplicate_group)
        self._session.flush()
        for photo in photos:
            self._session.add(
                DuplicateGroupPhoto(
                    duplicate_group_id=duplicate_group.id,
                    photo_id=photo.id,
                ),
            )


def _same_exact_hash(first_photo: Photo, second_photo: Photo) -> bool:
    return (
        first_photo.sha256 is not None
        and first_photo.sha256 != ""
        and first_photo.sha256 == second_photo.sha256
    )
