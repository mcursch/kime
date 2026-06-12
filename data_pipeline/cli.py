"""
data_pipeline/cli.py

Top-level CLI dispatcher for the Kime data pipeline.

Usage
-----
  python -m data_pipeline.cli review [--email REVIEWER] [--mp4]
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_pipeline",
        description="Kime data-pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- review sub-command ---
    review_parser = sub.add_parser(
        "review",
        help="Interactively review candidate skeleton files for approval.",
    )
    review_parser.add_argument(
        "--email",
        dest="reviewer_email",
        default="reviewer@example.com",
        metavar="EMAIL",
        help="Reviewer e-mail address recorded in manifest entries.",
    )
    review_parser.add_argument(
        "--mp4",
        dest="save_mp4",
        action="store_true",
        default=False,
        help=(
            "Save an MP4 preview to data/review_previews/ instead of "
            "rendering in the terminal."
        ),
    )
    review_parser.add_argument(
        "--landmarks-dir",
        dest="landmarks_dir",
        default=None,
        metavar="PATH",
        help="Override the default data/landmarks/ search directory.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "review":
        from data_pipeline.review import run_review
        from pathlib import Path

        landmarks_dir = Path(args.landmarks_dir) if args.landmarks_dir else None
        run_review(
            reviewer_email=args.reviewer_email,
            save_mp4=args.save_mp4,
            landmarks_dir=landmarks_dir,
        )
    else:
        parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
