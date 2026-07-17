from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from lensmind import app
from lensmind.app import build_parser, main
from lensmind.db.repository import PhotoRepository, initialize_sqlite


def test_parser_accepts_log_level() -> None:
    args = build_parser().parse_args(["--log-level", "DEBUG"])

    assert args.log_level == "DEBUG"


def test_evaluate_search_command_saves_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluation_path = tmp_path / "eval.json"
    output_path = tmp_path / "results.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "beach",
                        "text": "sunset beach",
                        "relevant_photo_ids": [10],
                    },
                ],
                "duplicate": {
                    "relevant_pairs": [[1, 2]],
                },
                "blur": {
                    "relevant_photo_ids": [5],
                },
            },
        ),
    )

    class FakeFaissPhotoSearchService:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def search_photos(self, query: str, limit: int) -> list[SimpleNamespace]:
            assert query == "sunset beach"
            assert limit == 1
            return [SimpleNamespace(photo_id=10, score=0.99)]

    monkeypatch.setattr(app, "FaissPhotoSearchService", FakeFaissPhotoSearchService)
    monkeypatch.setattr(
        app,
        "_predicted_duplicate_pairs",
        lambda session_factory: [(1, 2)],
    )
    monkeypatch.setattr(app, "_predicted_blurry_photo_ids", lambda session_factory: [5])

    database_path = tmp_path / "lensmind.sqlite3"
    exit_code = main(
        [
            "evaluate-search",
            "--eval-data",
            str(evaluation_path),
            "--output",
            str(output_path),
            "--top-k",
            "1",
            "--database",
            str(database_path),
            "--index-dir",
            str(tmp_path / "faiss"),
            "--model-name",
            "model",
            "--model-config",
            "config",
        ],
    )

    assert exit_code == 0
    metrics = json.loads(output_path.read_text())["metrics"]
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["duplicate_precision"] == 1.0
    assert metrics["duplicate_recall"] == 1.0
    assert metrics["duplicate_f1"] == 1.0
    assert metrics["blur_precision"] == 1.0
    assert metrics["blur_recall"] == 1.0
    assert metrics["blur_f1"] == 1.0
    assert "average_search_latency_ms" in metrics
    assert json.loads(output_path.read_text())["duplicate"] == {
        "metrics": {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        },
        "predicted_pairs": [[1, 2]],
        "relevant_pairs": [[1, 2]],
    }
    assert json.loads(output_path.read_text())["blur"] == {
        "metrics": {
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        },
        "predicted_photo_ids": [5],
        "relevant_photo_ids": [5],
    }
    with initialize_sqlite(database_path)() as session:
        runs = PhotoRepository(session).list_evaluation_runs()

    assert len(runs) == 1
    assert runs[0].report["metrics"] == json.loads(output_path.read_text())["metrics"]
