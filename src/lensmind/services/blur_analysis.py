from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class BlurThresholds:
    blurry: float = 100.0
    sharp: float = 500.0

    def __post_init__(self) -> None:
        if self.sharp <= self.blurry:
            msg = "sharp threshold must be greater than blurry threshold"
            raise ValueError(msg)


@dataclass(frozen=True)
class BlurAnalysisResult:
    path: Path
    raw_score: float | None
    display_score: float | None
    thresholds: BlurThresholds
    resized_for_analysis: bool
    error: str | None = None


class BlurAnalysisService:
    def __init__(
        self,
        thresholds: BlurThresholds | None = None,
        max_dimension: int = 1200,
    ) -> None:
        self._thresholds = thresholds or BlurThresholds()
        self._max_dimension = max_dimension

    def analyze(self, path: Path | str) -> BlurAnalysisResult:
        image_path = Path(path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return BlurAnalysisResult(
                path=image_path,
                raw_score=None,
                display_score=None,
                thresholds=self._thresholds,
                resized_for_analysis=False,
                error="Unable to read image",
            )

        resized_image, resized = self._resize_for_analysis(image)
        raw_score = float(cv2.Laplacian(resized_image, cv2.CV_64F).var())

        return BlurAnalysisResult(
            path=image_path,
            raw_score=raw_score,
            display_score=self._normalize(raw_score),
            thresholds=self._thresholds,
            resized_for_analysis=resized,
        )

    def _resize_for_analysis(self, image: object) -> tuple[object, bool]:
        height, width = image.shape[:2]
        largest_dimension = max(width, height)
        if largest_dimension <= self._max_dimension:
            return image, False

        scale = self._max_dimension / largest_dimension
        resized_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA), True

    def _normalize(self, raw_score: float) -> float:
        normalized = (
            (raw_score - self._thresholds.blurry)
            / (self._thresholds.sharp - self._thresholds.blurry)
        ) * 100
        return max(0.0, min(100.0, normalized))
