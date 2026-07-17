from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, ImageOps, UnidentifiedImageError

from lensmind.services.embeddings import EmbeddingProvider, EmbeddingResult

T = TypeVar("T")

DEFAULT_OPENCLIP_MODEL = "ViT-B-32"
# Standard OpenCLIP ViT-B-32 checkpoint with broad LAION-2B coverage.
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_EMBEDDING_BATCH_SIZE = 8


class EmbeddingConfigurationError(ValueError):
    pass


class EmbeddingModelLoadError(RuntimeError):
    pass


class OpenCLIPEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = DEFAULT_OPENCLIP_MODEL,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
        device: str = "auto",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        *,
        torch_module: Any | None = None,
        open_clip_module: Any | None = None,
    ) -> None:
        if batch_size < 1:
            msg = "batch_size must be at least 1"
            raise EmbeddingConfigurationError(msg)
        if device not in {"auto", "cpu", "mps"}:
            msg = f"unsupported device: {device}"
            raise EmbeddingConfigurationError(msg)

        self.model_name = model_name
        self.pretrained = pretrained
        self.requested_device = device
        self.batch_size = batch_size
        self._torch = torch_module
        self._open_clip = open_clip_module
        self._device: str | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def selected_device(self) -> str:
        torch = self._get_torch()
        if self.requested_device == "auto":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if self.requested_device == "mps" and not torch.backends.mps.is_available():
            msg = "mps requested but unavailable"
            raise EmbeddingConfigurationError(msg)
        return self.requested_device

    def embed_images(self, paths: list[Path]) -> list[EmbeddingResult]:
        self._ensure_loaded()
        results: list[EmbeddingResult] = []

        for batch in _batches(paths, self.batch_size):
            valid_items: list[tuple[int, Any]] = []
            batch_results = [
                self._empty_result(error=None)
                for _path in batch
            ]
            for index, path in enumerate(batch):
                tensor, error = self._load_image_tensor(path)
                if error is not None:
                    batch_results[index] = self._empty_result(error=error)
                else:
                    valid_items.append((index, tensor))

            if valid_items:
                embeddings = self._encode_image_batch(
                    [tensor for _index, tensor in valid_items],
                )
                for embedding_index, (result_index, _tensor) in enumerate(valid_items):
                    batch_results[result_index] = self._result_from_tensor(
                        embeddings[embedding_index],
                    )
            results.extend(batch_results)

        return results

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        self._ensure_loaded()
        results: list[EmbeddingResult] = []

        for batch in _batches(texts, self.batch_size):
            valid_items: list[tuple[int, str]] = []
            batch_results = [self._empty_result(error=None) for _text in batch]
            for index, text in enumerate(batch):
                if not text.strip():
                    batch_results[index] = self._empty_result(error="empty text")
                else:
                    valid_items.append((index, text))

            if valid_items:
                embeddings = self._encode_text_batch(
                    [text for _index, text in valid_items],
                )
                for embedding_index, (result_index, _text) in enumerate(valid_items):
                    batch_results[result_index] = self._result_from_tensor(
                        embeddings[embedding_index],
                    )
            results.extend(batch_results)

        return results

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        torch = self._get_torch()
        open_clip = self._get_open_clip()
        self._device = self.selected_device()
        try:
            model, _train_preprocess, eval_preprocess = (
                open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self._device,
                )
            )
            model.eval()
            self._model = model
            self._preprocess = eval_preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        except Exception as error:
            msg = f"failed to load OpenCLIP model: {error}"
            raise EmbeddingModelLoadError(msg) from error

        self._torch = torch

    def _load_image_tensor(self, path: Path) -> tuple[Any | None, str | None]:
        if not path.exists():
            return None, "missing file"
        try:
            with Image.open(path) as image:
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                return self._preprocess(normalized), None
        except (OSError, UnidentifiedImageError) as error:
            return None, str(error)

    def _encode_image_batch(self, tensors: Sequence[Any]) -> Any:
        torch = self._get_torch()
        with torch.no_grad():
            batch = torch.stack(list(tensors)).to(self._device)
            return self._normalize(self._model.encode_image(batch))

    def _encode_text_batch(self, texts: list[str]) -> Any:
        torch = self._get_torch()
        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self._device)
            return self._normalize(self._model.encode_text(tokens))

    def _normalize(self, embeddings: Any) -> Any:
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def _result_from_tensor(self, tensor: Any) -> EmbeddingResult:
        values = tuple(float(value) for value in tensor.detach().cpu().tolist())
        return EmbeddingResult(
            vector=values,
            dimension=len(values),
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self._device or "cpu",
        )

    def _empty_result(self, error: str | None) -> EmbeddingResult:
        return EmbeddingResult(
            vector=None,
            dimension=0,
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self._device or self.selected_device(),
            error=error,
        )

    def _get_torch(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _get_open_clip(self) -> Any:
        if self._open_clip is None:
            import open_clip

            self._open_clip = open_clip
        return self._open_clip


def _batches(items: list[T], batch_size: int) -> list[list[T]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]
