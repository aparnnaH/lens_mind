from __future__ import annotations

from pathlib import Path

from lensmind.db.repository import PhotoData, PhotoRepository, initialize_sqlite
from lensmind.services.duplicate_detection import DuplicateDetectionService
from lensmind.services.photo_ranking import (
    ExposureClipping,
    PhotoRankingService,
)


def test_ranks_higher_blur_score_and_resolution_above_weaker_photo(
    tmp_path: Path,
) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        weaker = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/weaker.jpg",
                filename="weaker.jpg",
                file_size=100,
                width=1200,
                height=800,
                blur_score=50.0,
            ),
        )
        stronger = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/stronger.jpg",
                filename="stronger.jpg",
                file_size=100,
                width=4000,
                height=3000,
                blur_score=500.0,
            ),
        )

        ranks = PhotoRankingService(session).rank_photos()

    assert [rank.photo_id for rank in ranks] == [stronger.id, weaker.id]
    assert ranks[0].score > ranks[1].score
    assert ranks[0].reasons == (
        "Higher blur score",
        "Higher resolution",
        "No duplicate preference",
        "No exposure clipping data",
    )


def test_duplicate_group_preference_adjusts_scores(tmp_path: Path) -> None:
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
                width=3000,
                height=2000,
                blur_score=300.0,
            ),
        )
        second = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/second.jpg",
                filename="second.jpg",
                file_size=100,
                sha256=duplicate_hash,
                width=3000,
                height=2000,
                blur_score=300.0,
            ),
        )
        DuplicateDetectionService(session).rebuild_duplicate_groups()
        group = repository.list_duplicate_groups()[0]
        repository.select_preferred_duplicate_photo(group.id, second.id)

        ranks = PhotoRankingService(session).rank_photos()

    assert [rank.photo_id for rank in ranks] == [second.id, first.id]
    assert "Preferred duplicate" in ranks[0].reasons
    assert "Not preferred duplicate" in ranks[1].reasons
    assert ranks[0].score - ranks[1].score == 35.0


def test_exposure_clipping_penalizes_when_available(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        repository = PhotoRepository(session)
        clean = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/clean.jpg",
                filename="clean.jpg",
                file_size=100,
                width=4000,
                height=3000,
                blur_score=500.0,
            ),
        )
        clipped = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/clipped.jpg",
                filename="clipped.jpg",
                file_size=100,
                width=4000,
                height=3000,
                blur_score=500.0,
            ),
        )

        ranks = PhotoRankingService(
            session,
            exposure_clipping_by_photo_id={
                clean.id: ExposureClipping(),
                clipped.id: ExposureClipping(
                    shadow_fraction=0.10,
                    highlight_fraction=0.25,
                ),
            },
        ).rank_photos()

    assert [rank.photo_id for rank in ranks] == [clean.id, clipped.id]
    assert ranks[0].score == 70.0
    assert ranks[1].score == 63.0
    assert "No exposure clipping" in ranks[0].reasons
    assert "Exposure clipping" in ranks[1].reasons


def test_missing_blur_and_resolution_use_neutral_reasons(tmp_path: Path) -> None:
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    with session_factory() as session:
        photo = PhotoRepository(session).add_or_update_photo(
            PhotoData(
                original_path="/photos/missing.jpg",
                filename="missing.jpg",
                file_size=100,
            ),
        )

        rank = PhotoRankingService(session).rank_photos()[0]

    assert rank.photo_id == photo.id
    assert rank.score == 35.0
    assert rank.reasons == (
        "No blur score",
        "No resolution",
        "No duplicate preference",
        "No exposure clipping data",
    )
