"""
Command-line interface for the Kime data pipeline.

Usage
-----
    python -m data_pipeline.cli fetch \\
        --sources clips.yaml \\
        [--staging-dir data/raw] \\
        [--max-duration 60] \\
        [--n-samples 10] \\
        [--rejection-threshold 0.5] \\
        [--log-level INFO]

The ``fetch`` sub-command:
1. Downloads every URL listed in *sources* via :mod:`data_pipeline.fetcher`.
2. Runs the single-person quality filter via :mod:`data_pipeline.filter`.
Accepted clips remain in ``<staging-dir>/<label>/``; rejected clips are
moved to ``<staging-dir>/rejected/<label>/``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from data_pipeline.fetcher import fetch_all
from data_pipeline.filter import filter_all


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    """Entry point for ``fetch`` sub-command."""
    staging = pathlib.Path(args.staging_dir)

    # ---- Step 1: download ------------------------------------------------
    downloaded = fetch_all(
        sources_file=args.sources,
        staging_dir=staging,
        max_duration=args.max_duration,
    )

    if not downloaded:
        logging.getLogger(__name__).warning(
            "No clips were downloaded. Check your sources file and network."
        )
        # Return 0 — this is not a crash; it may simply be an empty sources
        # file or all URLs were skipped due to 404s.
        return 0

    # ---- Step 2: filter --------------------------------------------------
    accepted, rejected = filter_all(
        staging_dir=staging,
        n_samples=args.n_samples,
        rejection_threshold=args.rejection_threshold,
        model_path=args.pose_model or None,
    )

    print(
        f"\nFetch summary:\n"
        f"  Downloaded : {len(downloaded)}\n"
        f"  Accepted   : {len(accepted)}\n"
        f"  Rejected   : {len(rejected)}\n"
        f"  Staging dir: {staging.resolve()}\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.cli",
        description="Kime data-pipeline tools.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------
    fetch_p = subparsers.add_parser(
        "fetch",
        help="Download clips and apply the single-person quality filter.",
        description=(
            "Download clips listed in a YAML/JSON sources file, trim them "
            "to --max-duration seconds, then reject clips that contain zero "
            "or more than one visible person."
        ),
    )
    fetch_p.add_argument(
        "--sources",
        required=True,
        metavar="FILE",
        help="YAML or JSON file mapping technique labels to lists of URLs.",
    )
    fetch_p.add_argument(
        "--staging-dir",
        default="data/raw",
        metavar="DIR",
        help="Root directory for downloaded clips (default: data/raw).",
    )
    fetch_p.add_argument(
        "--max-duration",
        type=int,
        default=60,
        metavar="SECS",
        help="Maximum clip duration in seconds; longer clips are trimmed (default: 60).",
    )
    fetch_p.add_argument(
        "--n-samples",
        type=int,
        default=10,
        metavar="N",
        help="Number of frames to sample per clip for pose filtering (default: 10).",
    )
    fetch_p.add_argument(
        "--rejection-threshold",
        type=float,
        default=0.5,
        metavar="FRAC",
        help=(
            "Fraction of sampled frames that must trigger a condition for "
            "the clip to be rejected (default: 0.5)."
        ),
    )
    fetch_p.add_argument(
        "--pose-model",
        default="",
        metavar="PATH",
        help=(
            "Path to a MediaPipe PoseLandmarker .task model bundle.  "
            "If omitted, falls back to the KIME_POSE_MODEL env-var then "
            "auto-downloads the lite model to ~/.cache/kime/."
        ),
    )
    fetch_p.set_defaults(func=cmd_fetch)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
