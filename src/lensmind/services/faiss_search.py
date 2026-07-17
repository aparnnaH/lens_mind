from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import PhotoRepository, StoredPhotoEmbeddingData
from lensmind.services.embeddings import EmbeddingProvider
from lensmind.services.openclip_embeddings import OpenCLIPEmbeddingProvider


@dataclass(frozen=True)
class FaissIndexStatus:
    index_exists: bool
    mapping_exists: bool
    stale: bool
    reason: str | None = None


@dataclass(frozen=True)
class FaissSearchResult:
    photo_id: int
    score: float


class FaissIndexError(RuntimeError):
    pass


class StaleFaissIndexError(FaissIndexError):
    pass


class FaissPhotoSearchService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        index_dir: Path | str,
        *,
        model_name: str,
        model_config: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._index_dir = Path(index_dir)
        self._model_name = model_name
        self._model_config = model_config
        self._embedding_provider = embedding_provider or OpenCLIPEmbeddingProvider(
            model_name=model_name,
            pretrained=model_config,
        )

    @property
    def index_path(self) -> Path:
        return self._index_dir / "photo_embeddings.faiss"

    @property
    def mapping_path(self) -> Path:
        return self._index_dir / "photo_embeddings.mapping.json"

    def build_index(self) -> FaissIndexStatus:
        embeddings = self._list_embeddings()
        vectors = _vectors_from_embeddings(embeddings)
        faiss = _get_faiss()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.mapping_path.write_text(
            json.dumps(
                {
                    "model_name": self._model_name,
                    "model_config": self._model_config,
                    "dimension": int(vectors.shape[1]),
                    "photo_ids": [embedding.photo_id for embedding in embeddings],
                    "signature": _signature(embeddings),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return self.index_status()

    def index_status(self) -> FaissIndexStatus:
        if not self.index_path.exists() and not self.mapping_path.exists():
            return FaissIndexStatus(
                index_exists=False,
                mapping_exists=False,
                stale=True,
                reason="missing index and mapping",
            )
        if not self.index_path.exists():
            return FaissIndexStatus(
                index_exists=False,
                mapping_exists=True,
                stale=True,
                reason="missing index",
            )
        if not self.mapping_path.exists():
            return FaissIndexStatus(
                index_exists=True,
                mapping_exists=False,
                stale=True,
                reason="missing mapping",
            )

        mapping = self._load_mapping()
        if mapping.get("model_name") != self._model_name:
            return self._stale_status("model name mismatch")
        if mapping.get("model_config") != self._model_config:
            return self._stale_status("model config mismatch")
        if mapping.get("signature") != _signature(self._list_embeddings()):
            return self._stale_status("embedding data changed")
        return FaissIndexStatus(
            index_exists=True,
            mapping_exists=True,
            stale=False,
        )

    def search(
        self,
        query_vector: tuple[float, ...] | list[float] | np.ndarray,
        *,
        top_k: int = 10,
    ) -> list[FaissSearchResult]:
        if top_k < 1:
            msg = "top_k must be at least 1"
            raise ValueError(msg)

        status = self.index_status()
        if status.stale:
            msg = status.reason or "stale index"
            raise StaleFaissIndexError(msg)

        mapping = self._load_mapping()
        photo_ids = [int(photo_id) for photo_id in mapping["photo_ids"]]
        if not photo_ids:
            return []

        dimension = int(mapping["dimension"])
        query = _normalized_query(query_vector, dimension)
        faiss = _get_faiss()
        index = faiss.read_index(str(self.index_path))
        scores, indices = index.search(query, min(top_k, len(photo_ids)))
        return [
            FaissSearchResult(
                photo_id=photo_ids[int(index_position)],
                score=float(score),
            )
            for score, index_position in zip(scores[0], indices[0], strict=False)
            if int(index_position) >= 0
        ]

    def search_photos(self, query: str, limit: int) -> list[FaissSearchResult]:
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)
        if limit < 1:
            msg = "limit must be at least 1"
            raise ValueError(msg)

        status = self.index_status()
        if status.stale:
            msg = status.reason or "stale index"
            raise StaleFaissIndexError(msg)

        try:
            embedding = self._embedding_provider.embed_texts([query])[0]
        except Exception as error:
            msg = f"failed to generate text embedding: {error}"
            raise FaissIndexError(msg) from error

        if embedding.error is not None:
            msg = f"failed to generate text embedding: {embedding.error}"
            raise FaissIndexError(msg)
        if embedding.vector is None:
            msg = "failed to generate text embedding: missing vector"
            raise FaissIndexError(msg)

        return self.search(embedding.vector, top_k=limit)

    def _list_embeddings(self) -> list[StoredPhotoEmbeddingData]:
        with self._session_factory() as session:
            return PhotoRepository(session).list_stored_photo_embeddings(
                model_name=self._model_name,
                model_config=self._model_config,
            )

    def _load_mapping(self) -> dict[str, Any]:
        return json.loads(self.mapping_path.read_text())

    def _stale_status(self, reason: str) -> FaissIndexStatus:
        return FaissIndexStatus(
            index_exists=True,
            mapping_exists=True,
            stale=True,
            reason=reason,
        )


def _vectors_from_embeddings(
    embeddings: list[StoredPhotoEmbeddingData],
) -> np.ndarray:
    if not embeddings:
        msg = "no stored embeddings available"
        raise FaissIndexError(msg)

    dimension = embeddings[0].vector_dimension
    vectors = np.vstack(
        [
            _vector_from_bytes(embedding.embedding_data, embedding.vector_dimension)
            for embedding in embeddings
        ],
    )
    if any(embedding.vector_dimension != dimension for embedding in embeddings):
        msg = "embedding dimensions do not match"
        raise FaissIndexError(msg)
    return _normalize(vectors)


def _vector_from_bytes(data: bytes, dimension: int) -> np.ndarray:
    values = array("f")
    values.frombytes(data)
    if len(values) != dimension:
        msg = "embedding byte length does not match vector dimension"
        raise FaissIndexError(msg)
    return np.asarray(values, dtype=np.float32)


def _normalized_query(
    query_vector: tuple[float, ...] | list[float] | np.ndarray,
    dimension: int,
) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != dimension:
        msg = "query vector dimension does not match index"
        raise ValueError(msg)
    return _normalize(query)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        msg = "embedding vectors must be non-zero"
        raise FaissIndexError(msg)
    return vectors / norms


def _signature(
    embeddings: list[StoredPhotoEmbeddingData],
) -> list[dict[str, str | int]]:
    return [
        {
            "id": embedding.id,
            "photo_id": embedding.photo_id,
            "generated_at": embedding.generated_at.isoformat(),
        }
        for embedding in embeddings
    ]


def _get_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        msg = "faiss-cpu is required for FAISS photo search"
        raise FaissIndexError(msg) from error
    return faiss
