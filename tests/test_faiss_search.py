from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from lensmind.db.repository import (
    PhotoData,
    PhotoEmbeddingData,
    PhotoRepository,
    initialize_sqlite,
)
from lensmind.services.embeddings import EmbeddingResult
from lensmind.services.faiss_search import (
    FaissIndexError,
    FaissPhotoSearchService,
    StaleFaissIndexError,
)

MODEL_NAME = "ViT-B-32"
MODEL_CONFIG = "test-pretrained"


def test_builds_persists_and_searches_photo_embedding_index(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    photo_ids = seed_embeddings(
        session_factory,
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.8, 0.6, 0.0),
        ],
    )
    service = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
    )

    status = service.build_index()
    results = service.search((1.0, 0.0, 0.0), top_k=2)
    reloaded_results = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
    ).search((1.0, 0.0, 0.0), top_k=2)

    assert status.stale is False
    assert service.index_path.exists()
    assert service.mapping_path.exists()
    assert [result.photo_id for result in results] == [photo_ids[0], photo_ids[2]]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)
    assert reloaded_results == results


def test_search_photos_embeds_text_query_and_returns_ranked_results(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    photo_ids = seed_embeddings(
        session_factory,
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.8, 0.6, 0.0),
        ],
    )
    embedding_provider = FakeTextEmbeddingProvider((1.0, 0.0, 0.0))
    service = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
        embedding_provider=embedding_provider,
    )
    service.build_index()

    results = service.search_photos("sunset beach", limit=2)

    assert embedding_provider.queries == ["sunset beach"]
    assert [result.photo_id for result in results] == [photo_ids[0], photo_ids[2]]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8])


def test_search_photos_handles_blank_query_and_model_failure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    seed_embeddings(session_factory, [(1.0, 0.0, 0.0)])
    service = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
        embedding_provider=FailingTextEmbeddingProvider(),
    )
    service.build_index()

    with pytest.raises(ValueError, match="query must not be blank"):
        service.search_photos("   ", limit=1)
    with pytest.raises(FaissIndexError, match="failed to generate text embedding"):
        service.search_photos("mountains", limit=1)


def test_detects_missing_and_stale_index(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    seed_embeddings(session_factory, [(1.0, 0.0, 0.0)])
    service = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
    )

    missing_status = service.index_status()
    service.build_index()

    with session_factory() as session:
        repository = PhotoRepository(session)
        photo = repository.add_or_update_photo(
            PhotoData(
                original_path="/photos/new.jpg",
                filename="new.jpg",
                file_size=100,
                sha256="b" * 64,
            ),
        )
        repository.save_photo_embedding(
            PhotoEmbeddingData(
                photo_id=photo.id,
                model_name=MODEL_NAME,
                model_config=MODEL_CONFIG,
                vector_dimension=3,
                embedding_data=vector_bytes((0.0, 1.0, 0.0)),
            ),
        )

    stale_status = service.index_status()

    assert missing_status.stale is True
    assert missing_status.reason == "missing index and mapping"
    assert stale_status.stale is True
    assert stale_status.reason == "embedding data changed"
    with pytest.raises(StaleFaissIndexError, match="embedding data changed"):
        service.search((1.0, 0.0, 0.0), top_k=1)


def test_search_photos_handles_stale_and_empty_index(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    session_factory = initialize_sqlite(tmp_path / "lensmind.db")
    embedding_provider = FakeTextEmbeddingProvider((1.0, 0.0, 0.0))
    service = FaissPhotoSearchService(
        session_factory,
        tmp_path / "faiss",
        model_name=MODEL_NAME,
        model_config=MODEL_CONFIG,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(StaleFaissIndexError, match="missing index and mapping"):
        service.search_photos("sunset", limit=1)
    assert embedding_provider.queries == []

    create_empty_index(service)

    assert service.search_photos("sunset", limit=3) == []
    assert embedding_provider.queries == ["sunset"]


def seed_embeddings(session_factory, vectors: list[tuple[float, ...]]) -> list[int]:
    photo_ids: list[int] = []
    with session_factory() as session:
        repository = PhotoRepository(session)
        for index, vector in enumerate(vectors):
            photo = repository.add_or_update_photo(
                PhotoData(
                    original_path=f"/photos/{index}.jpg",
                    filename=f"{index}.jpg",
                    file_size=100,
                    sha256=f"{index}" * 64,
                ),
            )
            repository.save_photo_embedding(
                PhotoEmbeddingData(
                    photo_id=photo.id,
                    model_name=MODEL_NAME,
                    model_config=MODEL_CONFIG,
                    vector_dimension=len(vector),
                    embedding_data=vector_bytes(vector),
                ),
            )
            photo_ids.append(photo.id)
    return photo_ids


def vector_bytes(vector: tuple[float, ...]) -> bytes:
    return array("f", vector).tobytes()


def create_empty_index(service: FaissPhotoSearchService) -> None:
    import faiss

    service.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss.IndexFlatIP(3), str(service.index_path))
    service.mapping_path.write_text(
        """
{
  "dimension": 3,
  "model_config": "test-pretrained",
  "model_name": "ViT-B-32",
  "photo_ids": [],
  "signature": []
}
""".strip(),
    )


class FakeTextEmbeddingProvider:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self._vector = vector
        self.queries: list[str] = []

    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
        return []

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        self.queries.extend(texts)
        return [
            EmbeddingResult(
                vector=self._vector,
                dimension=len(self._vector),
                model_name=MODEL_NAME,
                pretrained=MODEL_CONFIG,
                device="cpu",
            )
            for _text in texts
        ]


class FailingTextEmbeddingProvider:
    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
        return []

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        raise RuntimeError("boom")
