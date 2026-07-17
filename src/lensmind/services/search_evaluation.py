from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeVar, cast

EVALUATION_DATA_VERSION = 1
T = TypeVar("T")


class RankedSearchResult(Protocol):
    @property
    def photo_id(self) -> int:
        ...

    @property
    def score(self) -> float:
        ...


@dataclass(frozen=True)
class SemanticSearchEvaluationQuery:
    id: str
    text: str
    relevant_photo_ids: tuple[int, ...]


@dataclass(frozen=True)
class DuplicateEvaluationData:
    relevant_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BlurEvaluationData:
    relevant_photo_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationData:
    queries: tuple[SemanticSearchEvaluationQuery, ...]
    duplicate: DuplicateEvaluationData | None = None
    blur: BlurEvaluationData | None = None


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    precision: float
    recall: float
    f1: float


def load_evaluation_data(path: Path | str) -> EvaluationData:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        msg = "evaluation data must be a JSON object"
        raise ValueError(msg)
    _validate_data_version(data)

    queries = data.get("queries", [])
    if not isinstance(queries, list):
        msg = "evaluation data queries must be a list"
        raise ValueError(msg)

    return EvaluationData(
        queries=tuple(_query_from_json(query) for query in queries),
        duplicate=_duplicate_data_from_json(data.get("duplicate")),
        blur=_blur_data_from_json(data.get("blur")),
    )


def load_evaluation_queries(path: Path | str) -> list[SemanticSearchEvaluationQuery]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        msg = "evaluation data must be a JSON object"
        raise ValueError(msg)
    _validate_data_version(data)

    queries = data.get("queries")
    if not isinstance(queries, list):
        msg = "evaluation data must include a queries list"
        raise ValueError(msg)

    return [_query_from_json(query) for query in queries]


def evaluate_semantic_search(
    queries: Sequence[SemanticSearchEvaluationQuery],
    search: Callable[[str, int], Iterable[RankedSearchResult]],
    *,
    top_k: int,
    timer: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    query_reports: list[dict[str, object]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []
    latency_values_ms: list[float] = []

    for query in queries:
        started_at = timer()
        results = list(search(query.text, top_k))
        latency_ms = (timer() - started_at) * 1000
        retrieved_photo_ids = [result.photo_id for result in results[:top_k]]
        relevant_photo_ids = set(query.relevant_photo_ids)
        precision = precision_at_k(retrieved_photo_ids, relevant_photo_ids, top_k)
        recall = recall_at_k(retrieved_photo_ids, relevant_photo_ids, top_k)
        reciprocal_rank_value = reciprocal_rank(
            retrieved_photo_ids,
            relevant_photo_ids,
            top_k,
        )

        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_rank_values.append(reciprocal_rank_value)
        latency_values_ms.append(latency_ms)
        query_reports.append(
            {
                "id": query.id,
                "text": query.text,
                "relevant_photo_ids": list(query.relevant_photo_ids),
                "retrieved_photo_ids": retrieved_photo_ids,
                "latency_ms": latency_ms,
                "metrics": {
                    f"precision_at_{top_k}": precision,
                    f"recall_at_{top_k}": recall,
                    "reciprocal_rank": reciprocal_rank_value,
                },
                "results": [
                    {
                        "rank": rank,
                        "photo_id": result.photo_id,
                        "score": result.score,
                    }
                    for rank, result in enumerate(results[:top_k], start=1)
                ],
            },
        )

    return {
        "version": EVALUATION_DATA_VERSION,
        "top_k": top_k,
        "query_count": len(query_reports),
        "metrics": {
            f"precision_at_{top_k}": _mean(precision_values),
            f"recall_at_{top_k}": _mean(recall_values),
            "mean_reciprocal_rank": _mean(reciprocal_rank_values),
            "average_search_latency_ms": _mean(latency_values_ms),
        },
        "queries": query_reports,
    }


def add_duplicate_evaluation(
    report: dict[str, object],
    *,
    predicted_pairs: Iterable[Sequence[int]],
    relevant_pairs: Iterable[Sequence[int]],
) -> None:
    predicted = _normal_pair_set(predicted_pairs)
    relevant = _normal_pair_set(relevant_pairs)
    metrics = duplicate_metrics(predicted, relevant)
    report["metrics"] = {
        **_report_metrics(report),
        "duplicate_precision": metrics.precision,
        "duplicate_recall": metrics.recall,
        "duplicate_f1": metrics.f1,
    }
    report["duplicate"] = {
        "predicted_pairs": [list(pair) for pair in sorted(predicted)],
        "relevant_pairs": [list(pair) for pair in sorted(relevant)],
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
    }


def add_blur_evaluation(
    report: dict[str, object],
    *,
    predicted_photo_ids: Iterable[int],
    relevant_photo_ids: Iterable[int],
) -> None:
    predicted = set(predicted_photo_ids)
    relevant = set(relevant_photo_ids)
    metrics = blur_metrics(predicted, relevant)
    report["metrics"] = {
        **_report_metrics(report),
        "blur_precision": metrics.precision,
        "blur_recall": metrics.recall,
        "blur_f1": metrics.f1,
    }
    report["blur"] = {
        "predicted_photo_ids": sorted(predicted),
        "relevant_photo_ids": sorted(relevant),
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
    }


def duplicate_metrics(
    predicted_pairs: set[tuple[int, int]],
    relevant_pairs: set[tuple[int, int]],
) -> BinaryClassificationMetrics:
    return _binary_metrics(predicted_pairs, relevant_pairs)


def blur_metrics(
    predicted_photo_ids: set[int],
    relevant_photo_ids: set[int],
) -> BinaryClassificationMetrics:
    return _binary_metrics(predicted_photo_ids, relevant_photo_ids)


def save_evaluation_report(report: dict[str, object], path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def precision_at_k(
    retrieved_photo_ids: Sequence[int],
    relevant_photo_ids: set[int],
    k: int,
) -> float:
    _validate_metric_inputs(relevant_photo_ids, k)
    hits = _hit_count(retrieved_photo_ids, relevant_photo_ids, k)
    return hits / k


def recall_at_k(
    retrieved_photo_ids: Sequence[int],
    relevant_photo_ids: set[int],
    k: int,
) -> float:
    _validate_metric_inputs(relevant_photo_ids, k)
    hits = _hit_count(retrieved_photo_ids, relevant_photo_ids, k)
    return hits / len(relevant_photo_ids)


def reciprocal_rank(
    retrieved_photo_ids: Sequence[int],
    relevant_photo_ids: set[int],
    k: int,
) -> float:
    _validate_metric_inputs(relevant_photo_ids, k)
    for rank, photo_id in enumerate(retrieved_photo_ids[:k], start=1):
        if photo_id in relevant_photo_ids:
            return 1 / rank
    return 0.0


def _validate_data_version(data: dict[str, object]) -> None:
    if data.get("version") != EVALUATION_DATA_VERSION:
        msg = "evaluation data version must be 1"
        raise ValueError(msg)


def _query_from_json(data: object) -> SemanticSearchEvaluationQuery:
    if not isinstance(data, dict):
        msg = "each evaluation query must be a JSON object"
        raise ValueError(msg)

    query_id = data.get("id")
    text = data.get("text")
    relevant_photo_ids = data.get("relevant_photo_ids")
    if not isinstance(query_id, str) or not query_id.strip():
        msg = "each evaluation query must include a non-empty id"
        raise ValueError(msg)
    if not isinstance(text, str) or not text.strip():
        msg = f"evaluation query {query_id!r} must include non-empty text"
        raise ValueError(msg)
    if not isinstance(relevant_photo_ids, list):
        msg = f"evaluation query {query_id!r} must include relevant_photo_ids"
        raise ValueError(msg)

    parsed_photo_ids = tuple(
        _parse_photo_id(photo_id)
        for photo_id in relevant_photo_ids
    )
    if not parsed_photo_ids:
        msg = (
            f"evaluation query {query_id!r} must include at least one relevant photo id"
        )
        raise ValueError(msg)

    return SemanticSearchEvaluationQuery(
        id=query_id,
        text=text,
        relevant_photo_ids=tuple(dict.fromkeys(parsed_photo_ids)),
    )


def _duplicate_data_from_json(data: object) -> DuplicateEvaluationData | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        msg = "duplicate evaluation data must be a JSON object"
        raise ValueError(msg)
    relevant_pairs = data.get("relevant_pairs")
    if not isinstance(relevant_pairs, list):
        msg = "duplicate evaluation data must include relevant_pairs"
        raise ValueError(msg)
    pairs = tuple(_parse_photo_pair(pair) for pair in relevant_pairs)
    if not pairs:
        msg = "duplicate evaluation data must include at least one relevant pair"
        raise ValueError(msg)
    return DuplicateEvaluationData(relevant_pairs=tuple(dict.fromkeys(pairs)))


def _blur_data_from_json(data: object) -> BlurEvaluationData | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        msg = "blur evaluation data must be a JSON object"
        raise ValueError(msg)
    relevant_photo_ids = data.get("relevant_photo_ids")
    if not isinstance(relevant_photo_ids, list):
        msg = "blur evaluation data must include relevant_photo_ids"
        raise ValueError(msg)
    photo_ids = tuple(_parse_photo_id(photo_id) for photo_id in relevant_photo_ids)
    if not photo_ids:
        msg = "blur evaluation data must include at least one relevant photo id"
        raise ValueError(msg)
    return BlurEvaluationData(relevant_photo_ids=tuple(dict.fromkeys(photo_ids)))


def _parse_photo_pair(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        msg = "duplicate relevant pairs must be two-item photo id lists"
        raise ValueError(msg)
    first_photo_id = _parse_photo_id(value[0])
    second_photo_id = _parse_photo_id(value[1])
    if first_photo_id == second_photo_id:
        msg = "duplicate relevant pairs must contain two different photo ids"
        raise ValueError(msg)
    return _ordered_pair(first_photo_id, second_photo_id)


def _parse_photo_id(value: object) -> int:
    if type(value) is not int or value < 1:
        msg = "relevant photo ids must be positive integers"
        raise ValueError(msg)
    return value


def _binary_metrics(
    predicted: set[T],
    relevant: set[T],
) -> BinaryClassificationMetrics:
    if not relevant:
        msg = "relevant items must not be empty"
        raise ValueError(msg)
    true_positive_count = len(predicted & relevant)
    precision = true_positive_count / len(predicted) if predicted else 0.0
    recall = true_positive_count / len(relevant)
    f1 = 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return BinaryClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _normal_pair_set(pairs: Iterable[Sequence[int]]) -> set[tuple[int, int]]:
    return {_normal_pair(pair) for pair in pairs}


def _normal_pair(pair: Sequence[int]) -> tuple[int, int]:
    if len(pair) != 2:
        msg = "duplicate pairs must contain exactly two photo ids"
        raise ValueError(msg)
    first_photo_id, second_photo_id = pair
    if first_photo_id == second_photo_id:
        msg = "duplicate pairs must contain two different photo ids"
        raise ValueError(msg)
    return _ordered_pair(first_photo_id, second_photo_id)


def _ordered_pair(first_photo_id: int, second_photo_id: int) -> tuple[int, int]:
    if first_photo_id < second_photo_id:
        return first_photo_id, second_photo_id
    return second_photo_id, first_photo_id


def _report_metrics(report: dict[str, object]) -> dict[str, object]:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return cast(dict[str, object], metrics)


def _validate_metric_inputs(relevant_photo_ids: set[int], k: int) -> None:
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    if not relevant_photo_ids:
        msg = "relevant_photo_ids must not be empty"
        raise ValueError(msg)


def _hit_count(
    retrieved_photo_ids: Sequence[int],
    relevant_photo_ids: set[int],
    k: int,
) -> int:
    return len(set(retrieved_photo_ids[:k]) & relevant_photo_ids)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
