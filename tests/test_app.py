from lensmind.app import build_parser


def test_parser_accepts_log_level() -> None:
    args = build_parser().parse_args(["--log-level", "DEBUG"])

    assert args.log_level == "DEBUG"
