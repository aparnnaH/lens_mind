from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import select

from lensmind.db.models import DuplicateGroup, DuplicateGroupPhoto
from lensmind.db.repository import PhotoData, PhotoRepository, initialize_sqlite
from lensmind.services.duplicate_detection import DuplicateDetectionService


def test_duplicate_detection_groups_photos_by_matching_sha256(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    duplicate_hash = "a" * 64

    with session_factory() as session:
        repository = PhotoRepository(session)
        first = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/first.jpg",
                filename="first.jpg",
                file_size=100,
                sha256=duplicate_hash,
            ),
        )
        second = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/second.jpg",
                filename="second.jpg",
                file_size=100,
                sha256=duplicate_hash,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/unique.jpg",
                filename="unique.jpg",
                file_size=100,
                sha256="b" * 64,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/unhashed.jpg",
                filename="unhashed.jpg",
                file_size=100,
            ),
        )

        summary = DuplicateDetectionService(session).rebuild_duplicate_groups()
        groups = list(session.scalars(select(DuplicateGroup)))
        group_links = list(session.scalars(select(DuplicateGroupPhoto)))

    assert summary.groups_created == 1
    assert summary.duplicate_photos == 2
    assert len(groups) == 1
    assert groups[0].classification == "exact"
    assert groups[0].sha256 == duplicate_hash
    assert {link.photo_id for link in group_links} == {first.id, second.id}


def test_duplicate_detection_rebuilds_existing_groups(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/first.jpg",
                filename="first.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/second.jpg",
                filename="second.jpg",
                file_size=100,
                sha256="a" * 64,
            ),
        )
        service = DuplicateDetectionService(session)
        first_summary = service.rebuild_duplicate_groups()
        second_summary = service.rebuild_duplicate_groups()
        groups = list(session.scalars(select(DuplicateGroup)))
        group_links = list(session.scalars(select(DuplicateGroupPhoto)))

    assert first_summary.groups_created == 1
    assert second_summary.groups_created == 1
    assert len(groups) == 1
    assert len(group_links) == 2


def test_duplicate_detection_finds_likely_duplicates_with_perceptual_hashes(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    image_path = tmp_path / "base.png"
    resized_path = tmp_path / "resized.png"
    modified_path = tmp_path / "modified.png"
    _create_pattern_image(image_path)
    Image.open(image_path).resize((96, 96)).save(resized_path)
    modified = Image.open(image_path)
    draw = ImageDraw.Draw(modified)
    draw.rectangle((0, 0, 8, 8), fill="white")
    modified.save(modified_path)

    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(image_path),
                filename="base.png",
                file_size=image_path.stat().st_size,
                sha256="a" * 64,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(resized_path),
                filename="resized.png",
                file_size=resized_path.stat().st_size,
                sha256="b" * 64,
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path=str(modified_path),
                filename="modified.png",
                file_size=modified_path.stat().st_size,
                sha256="c" * 64,
            ),
        )

        summary = DuplicateDetectionService(
            session,
            distance_threshold=8,
        ).rebuild_duplicate_groups()
        groups = list(session.scalars(select(DuplicateGroup)))

    assert summary.exact_groups_created == 0
    assert summary.likely_groups_created >= 1
    assert {group.classification for group in groups} == {"likely"}
    assert {group.sha256 for group in groups} == {None}
    assert {group.distance_threshold for group in groups} == {8}


def test_duplicate_detection_threshold_is_configurable(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/first.jpg",
                filename="first.jpg",
                file_size=100,
                sha256="a" * 64,
                perceptual_hash="0000000000000000",
            ),
        )
        repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/second.jpg",
                filename="second.jpg",
                file_size=100,
                sha256="b" * 64,
                perceptual_hash="0000000000000003",
            ),
        )

        strict_summary = DuplicateDetectionService(
            session,
            distance_threshold=1,
        ).rebuild_duplicate_groups()
        relaxed_summary = DuplicateDetectionService(
            session,
            distance_threshold=2,
        ).rebuild_duplicate_groups()

    assert strict_summary.likely_groups_created == 0
    assert relaxed_summary.likely_groups_created == 1


def test_likely_duplicate_detection_does_not_merge_uncertain_chains(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")

    with session_factory() as session:
        repository = PhotoRepository(session)
        for filename, perceptual_hash in (
            ("first.jpg", "0000000000000000"),
            ("second.jpg", "0000000000000001"),
            ("third.jpg", "0000000000000003"),
        ):
            repository.add_or_update_photo(
                PhotoData(
                    original_path=f"/photos/{filename}",
                    filename=filename,
                    file_size=100,
                    sha256=filename * 8,
                    perceptual_hash=perceptual_hash,
                ),
            )

        summary = DuplicateDetectionService(
            session,
            distance_threshold=1,
        ).rebuild_duplicate_groups()
        groups = list(session.scalars(select(DuplicateGroup)))
        group_links = list(session.scalars(select(DuplicateGroupPhoto)))

    assert summary.likely_groups_created == 2
    assert all(group.classification == "likely" for group in groups)
    assert all(group.max_distance == 1 for group in groups)
    assert len(group_links) == 4


def _create_pattern_image(path: Path) -> None:
    image = Image.new("RGB", (128, 128), color="black")
    draw = ImageDraw.Draw(image)
    for index in range(0, 128, 16):
        draw.rectangle((index, 0, index + 7, 127), fill="white")
        draw.rectangle((0, index, 127, index + 7), fill="gray")
    image.save(path)
