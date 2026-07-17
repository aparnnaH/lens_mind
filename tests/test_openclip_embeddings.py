from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from lensmind.services.openclip_embeddings import (
    DEFAULT_OPENCLIP_MODEL,
    DEFAULT_OPENCLIP_PRETRAINED,
    EmbeddingConfigurationError,
    EmbeddingModelLoadError,
    OpenCLIPEmbeddingProvider,
)


def test_device_selection_uses_mps_when_available() -> None:
    provider = OpenCLIPEmbeddingProvider(torch_module=FakeTorch(mps_available=True))

    assert provider.selected_device() == "mps"


def test_device_selection_falls_back_to_cpu() -> None:
    provider = OpenCLIPEmbeddingProvider(torch_module=FakeTorch(mps_available=False))

    assert provider.selected_device() == "cpu"


def test_lazy_loading_and_normalized_text_output() -> None:
    fake_open_clip = FakeOpenCLIP()
    provider = OpenCLIPEmbeddingProvider(open_clip_module=fake_open_clip)

    assert provider.is_loaded is False

    result = provider.embed_texts(["sunset"])[0]

    assert provider.is_loaded is True
    assert fake_open_clip.load_count == 1
    assert result.error is None
    assert result.vector == pytest.approx((0.6, 0.8))
    assert result.dimension == 2
    assert result.model_name == DEFAULT_OPENCLIP_MODEL
    assert result.pretrained == DEFAULT_OPENCLIP_PRETRAINED
    assert result.device == "cpu"


def test_image_embeddings_accept_paths_and_return_normalized_vectors(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    provider = OpenCLIPEmbeddingProvider(open_clip_module=FakeOpenCLIP())

    result = provider.embed_images([image_path])[0]

    assert result.error is None
    assert result.vector == pytest.approx((0.6, 0.8))
    assert result.dimension == 2


def test_empty_text_is_rejected() -> None:
    provider = OpenCLIPEmbeddingProvider(open_clip_module=FakeOpenCLIP())

    result = provider.embed_texts(["   "])[0]

    assert result.vector is None
    assert result.dimension == 0
    assert result.error == "empty text"


def test_missing_image_returns_error(tmp_path: Path) -> None:
    provider = OpenCLIPEmbeddingProvider(open_clip_module=FakeOpenCLIP())

    result = provider.embed_images([tmp_path / "missing.jpg"])[0]

    assert result.vector is None
    assert result.dimension == 0
    assert result.error == "missing file"


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="batch_size"):
        OpenCLIPEmbeddingProvider(batch_size=0)


def test_unsupported_device_request_is_rejected() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="unsupported device"):
        OpenCLIPEmbeddingProvider(device="cuda")


def test_unavailable_mps_request_is_rejected() -> None:
    provider = OpenCLIPEmbeddingProvider(
        device="mps",
        torch_module=FakeTorch(mps_available=False),
    )

    with pytest.raises(EmbeddingConfigurationError, match="mps requested"):
        provider.selected_device()


def test_model_loading_failures_are_wrapped() -> None:
    provider = OpenCLIPEmbeddingProvider(open_clip_module=FailingOpenCLIP())

    with pytest.raises(EmbeddingModelLoadError, match="failed to load OpenCLIP"):
        provider.embed_texts(["hello"])


class FakeMPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = FakeMPS(mps_available)


class FakeTorch:
    def __init__(self, mps_available: bool) -> None:
        self.backends = FakeBackends(mps_available)


class FakeModel:
    def eval(self) -> None:
        return None

    def encode_text(self, _tokens):
        import torch

        return torch.tensor([[3.0, 4.0]])

    def encode_image(self, _batch):
        import torch

        return torch.tensor([[3.0, 4.0]])


class FakeOpenCLIP:
    def __init__(self) -> None:
        self.load_count = 0

    def create_model_and_transforms(self, model_name, pretrained, device):
        self.load_count += 1
        assert model_name == DEFAULT_OPENCLIP_MODEL
        assert pretrained == DEFAULT_OPENCLIP_PRETRAINED
        assert device == "cpu"
        return FakeModel(), object(), fake_preprocess

    def get_tokenizer(self, model_name):
        assert model_name == DEFAULT_OPENCLIP_MODEL
        return fake_tokenizer


class FailingOpenCLIP:
    def create_model_and_transforms(self, *_args, **_kwargs):
        raise RuntimeError("boom")


def fake_preprocess(_image):
    import torch

    return torch.ones(3)


def fake_tokenizer(texts):
    import torch

    return torch.ones((len(texts), 3))
