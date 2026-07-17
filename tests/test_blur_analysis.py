from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from lensmind.services.blur_analysis import BlurAnalysisService, BlurThresholds


def create_sharp_fixture(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    image = Image.new("L", size, color=0)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = 255 if (x // 8 + y // 8) % 2 == 0 else 0
    image.save(path)


def test_blur_analysis_scores_sharp_image_above_blurred_image(tmp_path: Path) -> None:
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"
    create_sharp_fixture(sharp_path)
    Image.open(sharp_path).filter(ImageFilter.GaussianBlur(radius=8)).save(
        blurred_path,
    )
    service = BlurAnalysisService()

    sharp_result = service.analyze(sharp_path)
    blurred_result = service.analyze(blurred_path)

    assert sharp_result.error is None
    assert blurred_result.error is None
    assert sharp_result.raw_score is not None
    assert blurred_result.raw_score is not None
    assert sharp_result.raw_score > blurred_result.raw_score
    assert sharp_result.display_score is not None
    assert blurred_result.display_score is not None
    assert sharp_result.display_score > blurred_result.display_score


def test_blur_analysis_resizes_large_images_before_analysis(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    create_sharp_fixture(image_path, size=(300, 200))

    result = BlurAnalysisService(max_dimension=100).analyze(image_path)

    assert result.error is None
    assert result.raw_score is not None
    assert result.resized_for_analysis is True


def test_blur_analysis_uses_configurable_thresholds(tmp_path: Path) -> None:
    image_path = tmp_path / "sharp.png"
    create_sharp_fixture(image_path)
    thresholds = BlurThresholds(blurry=10.0, sharp=20.0)

    result = BlurAnalysisService(thresholds=thresholds).analyze(image_path)

    assert result.thresholds == thresholds
    assert result.display_score == 100.0


def test_blur_analysis_handles_corrupted_files(tmp_path: Path) -> None:
    image_path = tmp_path / "bad.jpg"
    image_path.write_bytes(b"not an image")

    result = BlurAnalysisService().analyze(image_path)

    assert result.raw_score is None
    assert result.display_score is None
    assert result.error == "Unable to read image"
