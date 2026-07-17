from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...] | None
    dimension: int
    model_name: str
    pretrained: str
    device: str
    error: str | None = None


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        raise NotImplementedError
