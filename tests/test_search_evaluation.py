from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lensmind.services.search_evaluation import (
    SemanticSearchEvaluationQuery,
    add_blur_evaluation,
    add_duplicate_evaluation,
    blur_metrics,
    duplicate_metrics,
    evaluate_semantic_search,
    load_evaluation_data,
    load_evaluation_queries,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    save_evaluation_report,
)


def test_computes_precision_recall_and_reciprocal_rank_at_k() -> None:
    retrieved_photo_ids = [2, 3, 4]
    relevant_photo_ids = {1, 3, 5}

    assert precision_at_k(retrieved_photo_ids, relevant_photo_ids, 3) == pytest.approx(
        1 / 3,
    )
    assert recall_at_k(retrieved_photo_ids, relevant_photo_ids, 3) == pytest.approx(
        1 / 3,
    )
    assert reciprocal_rank(retrieved_photo_ids, relevant_photo_ids, 3) == pytest.approx(
        1 / 2,
    )


def test_evaluates_queries_and_saves_privacy_safe_json(tmp_path: Path) -> None:
    queries = [
        SemanticSearchEvaluationQuery(
            id="beach",
            text="sunset beach",
            relevant_photo_ids=(1, 3),
        ),
        SemanticSearchEvaluationQuery(
            id="mountain",
            text="snowy mountain",
            relevant_photo_ids=(5,),
        ),
    ]

    def search(text: str, top_k: int) -> list[SimpleNamespace]:
        assert top_k == 2
        if text == "sunset beach":
            return [
                SimpleNamespace(photo_id=3, score=0.9),
                SimpleNamespace(photo_id=2, score=0.4),
            ]
        return [
            SimpleNamespace(photo_id=4, score=0.8),
            SimpleNamespace(photo_id=5, score=0.7),
        ]

    timer_values = iter([0.0, 0.010, 0.010, 0.040])

    report = evaluate_semantic_search(
        queries,
        search,
        top_k=2,
        timer=lambda: next(timer_values),
    )
    output_path = tmp_path / "results.json"

    save_evaluation_report(report, output_path)
    saved_report = json.loads(output_path.read_text())

    assert saved_report["metrics"] == {
        "mean_reciprocal_rank": 0.75,
        "precision_at_2": 0.5,
        "recall_at_2": 0.75,
        "average_search_latency_ms": 20.0,
    }
    assert saved_report["queries"][0]["latency_ms"] == 10.0
    assert saved_report["queries"][0]["results"] == [
        {"photo_id": 3, "rank": 1, "score": 0.9},
        {"photo_id": 2, "rank": 2, "score": 0.4},
    ]
    assert "original_path" not in output_path.read_text()
    assert "filename" not in output_path.read_text()


def test_computes_duplicate_and_blur_precision_recall_f1() -> None:
    duplicate = duplicate_metrics(
        predicted_pairs={(1, 2), (2, 3)},
        relevant_pairs={(1, 2), (4, 5)},
    )
    blur = blur_metrics(
        predicted_photo_ids={1, 2, 3},
        relevant_photo_ids={2, 3, 4, 5},
    )

    assert duplicate.precision == pytest.approx(0.5)
    assert duplicate.recall == pytest.approx(0.5)
    assert duplicate.f1 == pytest.approx(0.5)
    assert blur.precision == pytest.approx(2 / 3)
    assert blur.recall == pytest.approx(0.5)
    assert blur.f1 == pytest.approx(4 / 7)


def test_adds_duplicate_and_blur_sections_to_report() -> None:
    report: dict[str, object] = {
        "version": 1,
        "top_k": 2,
        "query_count": 0,
        "metrics": {},
        "queries": [],
    }

    add_duplicate_evaluation(
        report,
        predicted_pairs=[(2, 1), (3, 4)],
        relevant_pairs=[(1, 2), (5, 6)],
    )
    add_blur_evaluation(
        report,
        predicted_photo_ids=[1, 2, 3],
        relevant_photo_ids=[2, 4],
    )

    assert report["metrics"] == {
        "duplicate_precision": 0.5,
        "duplicate_recall": 0.5,
        "duplicate_f1": 0.5,
        "blur_precision": pytest.approx(1 / 3),
        "blur_recall": 0.5,
        "blur_f1": 0.4,
    }
    assert report["duplicate"] == {
        "predicted_pairs": [[1, 2], [3, 4]],
        "relevant_pairs": [[1, 2], [5, 6]],
        "metrics": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
    }
    assert report["blur"] == {
        "predicted_photo_ids": [1, 2, 3],
        "relevant_photo_ids": [2, 4],
        "metrics": {
            "precision": pytest.approx(1 / 3),
            "recall": 0.5,
            "f1": 0.4,
        },
    }


def test_loads_small_evaluation_data_format(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "eval.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "city",
                        "text": "night street",
                        "relevant_photo_ids": [7, 7, 8],
                    },
                ],
            },
        ),
    )

    queries = load_evaluation_queries(evaluation_path)

    assert queries == [
        SemanticSearchEvaluationQuery(
            id="city",
            text="night street",
            relevant_photo_ids=(7, 8),
        ),
    ]


def test_loads_optional_duplicate_and_blur_ground_truth(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "eval.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [],
                "duplicate": {
                    "relevant_pairs": [[2, 1], [3, 4]],
                },
                "blur": {
                    "relevant_photo_ids": [8, 8, 9],
                },
            },
        ),
    )

    data = load_evaluation_data(evaluation_path)

    assert data.queries == ()
    assert data.duplicate is not None
    assert data.duplicate.relevant_pairs == ((1, 2), (3, 4))
    assert data.blur is not None
    assert data.blur.relevant_photo_ids == (8, 9)


def test_rejects_queries_without_relevant_photo_ids(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "eval.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "empty",
                        "text": "empty ground truth",
                        "relevant_photo_ids": [],
                    },
                ],
            },
        ),
    )

    with pytest.raises(ValueError, match="at least one relevant photo id"):
        load_evaluation_queries(evaluation_path)
