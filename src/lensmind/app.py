from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from lensmind.db.repository import PhotoRepository, initialize_sqlite
from lensmind.logging import configure_logging
from lensmind.services.blur_analysis import BlurThresholds
from lensmind.services.faiss_search import FaissPhotoSearchService
from lensmind.services.openclip_embeddings import (
    DEFAULT_OPENCLIP_MODEL,
    DEFAULT_OPENCLIP_PRETRAINED,
)
from lensmind.services.search_evaluation import (
    EVALUATION_DATA_VERSION,
    add_blur_evaluation,
    add_duplicate_evaluation,
    evaluate_semantic_search,
    load_evaluation_data,
    save_evaluation_report,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lensmind")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set application log verbosity.",
    )
    subparsers = parser.add_subparsers(dest="command")
    evaluation_parser = subparsers.add_parser(
        "evaluate-search",
        help="Evaluate semantic search quality and save JSON results.",
    )
    evaluation_parser.add_argument(
        "--eval-data",
        type=Path,
        required=True,
        help="Path to semantic search evaluation data JSON.",
    )
    evaluation_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where evaluation results JSON will be written.",
    )
    evaluation_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked search results to evaluate.",
    )
    evaluation_parser.add_argument(
        "--database",
        type=Path,
        default=Path.home() / ".lensmind" / "lensmind.sqlite3",
        help="Path to the LensMind SQLite database.",
    )
    evaluation_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path.home() / ".lensmind" / "faiss",
        help="Directory containing the FAISS semantic search index.",
    )
    evaluation_parser.add_argument(
        "--model-name",
        default=DEFAULT_OPENCLIP_MODEL,
        help="Embedding model name used by the FAISS index.",
    )
    evaluation_parser.add_argument(
        "--model-config",
        default=DEFAULT_OPENCLIP_PRETRAINED,
        help="Embedding model config used by the FAISS index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "evaluate-search":
        return _run_search_evaluation(args)

    logger.info("LensMind application starting")

    from lensmind.ui.shell import run_application

    return run_application([])


def _run_search_evaluation(args: argparse.Namespace) -> int:
    evaluation_data = load_evaluation_data(args.eval_data)
    args.database.expanduser().parent.mkdir(parents=True, exist_ok=True)
    session_factory = initialize_sqlite(args.database)

    report = (
        evaluate_semantic_search(
            evaluation_data.queries,
            FaissPhotoSearchService(
                session_factory,
                args.index_dir,
                model_name=args.model_name,
                model_config=args.model_config,
            ).search_photos,
            top_k=args.top_k,
        )
        if evaluation_data.queries
        else _empty_evaluation_report(args.top_k)
    )

    if evaluation_data.duplicate is not None:
        add_duplicate_evaluation(
            report,
            predicted_pairs=_predicted_duplicate_pairs(session_factory),
            relevant_pairs=evaluation_data.duplicate.relevant_pairs,
        )
    if evaluation_data.blur is not None:
        add_blur_evaluation(
            report,
            predicted_photo_ids=_predicted_blurry_photo_ids(session_factory),
            relevant_photo_ids=evaluation_data.blur.relevant_photo_ids,
    )

    save_evaluation_report(report, args.output)
    with session_factory() as session:
        PhotoRepository(session).record_evaluation_run(report)
    logger.info("Semantic search evaluation saved to %s", args.output)
    return 0


def _empty_evaluation_report(top_k: int) -> dict[str, object]:
    return {
        "version": EVALUATION_DATA_VERSION,
        "top_k": top_k,
        "query_count": 0,
        "metrics": {
            f"precision_at_{top_k}": 0.0,
            f"recall_at_{top_k}": 0.0,
            "mean_reciprocal_rank": 0.0,
            "average_search_latency_ms": 0.0,
        },
        "queries": [],
    }


def _predicted_duplicate_pairs(
    session_factory: sessionmaker[Session],
) -> list[tuple[int, int]]:
    with session_factory() as session:
        groups = PhotoRepository(session).list_duplicate_groups()
    pairs: list[tuple[int, int]] = []
    for group in groups:
        photo_ids = [photo.id for photo in group.photos]
        pairs.extend(
            _ordered_pair(first_photo_id, second_photo_id)
            for first_photo_id, second_photo_id in combinations(photo_ids, 2)
        )
    return pairs


def _ordered_pair(first_photo_id: int, second_photo_id: int) -> tuple[int, int]:
    if first_photo_id < second_photo_id:
        return first_photo_id, second_photo_id
    return second_photo_id, first_photo_id


def _predicted_blurry_photo_ids(
    session_factory: sessionmaker[Session],
) -> list[int]:
    with session_factory() as session:
        photos = PhotoRepository(session).list_blurry_photos(BlurThresholds().blurry)
    return [photo.id for photo in photos]
