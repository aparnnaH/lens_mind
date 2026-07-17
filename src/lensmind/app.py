from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from lensmind.logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lensmind")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set application log verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    logger.info("LensMind application starting")

    from lensmind.ui.shell import run_application

    return run_application([])
