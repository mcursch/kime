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

    python -m data_pipeline.cli extract \\
        --manifest data/staged/manifest.json \\
        [--landmarks-dir data/landmarks] \\
        [--dry-run]

    python -m data_pipeline.cli review \\
        [--email reviewer@example.com] \\
        [--mp4]

Sub-commands
------------
``fetch``
    Downloads every URL listed in *sources* via :mod:`data_pipeline.fetcher`
    and runs the single-person quality filter via :mod:`data_pipeline.filter`.
``extract``
    Runs skeleton extraction on every accepted clip listed in a manifest.
``review``
    Launches the interactive human-review loop for approving skeleton files.
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


def cmd_fetch(args: argparse.Namespace) -> None:
    """Entry point for ``fetch`` sub-command.

    Calls ``sys.exit`` directly so that the process exits with the correct
    code.  This keeps the public ``main()`` return-value contract clean for
    other sub-commands that return an integer exit code.
    """
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
        # Exit 0 — this is not a crash; it may simply be an empty sources
        # file or all URLs were skipped due to 404s.
        sys.exit(0)

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
    sys.exit(0)


def cmd_extract(args: argparse.Namespace) -> int:
    """Entry point for ``extract`` sub-command."""
    from data_pipeline.extractor import extract_all, load_accepted_clips

    manifest_path = pathlib.Path(args.manifest)
    landmarks_dir = pathlib.Path(args.landmarks_dir)

    if args.dry_run:
        try:
            clips = load_accepted_clips(manifest_path)
        except FileNotFoundError:
            print(
                f"Manifest not found: {manifest_path}",
                file=sys.stderr,
            )
            return 1
        print(f"Dry run — would process {len(clips)} accepted clip(s):")
        for clip in clips:
            print(
                f"  {clip.get('clip_id', '?')}"
                f"  ({clip.get('technique', '?')})"
                f"  {clip.get('path', '')}"
            )
        return 0

    try:
        extract_all(
            manifest_path=manifest_path,
            landmarks_dir=landmarks_dir,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Entry point for ``review`` sub-command."""
    from data_pipeline.review import run_review

    run_review(
        reviewer_email=args.reviewer_email,
        save_mp4=args.save_mp4,
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
    subparsers.required = False

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

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------
    extract_p = subparsers.add_parser(
        "extract",
        help="Extract skeleton landmarks from accepted staged clips.",
        description=(
            "Read the clip manifest, run MediaPipe PoseLandmarker on every "
            "accepted clip, then write smoothed and normalised .npz files "
            "under --landmarks-dir."
        ),
    )
    extract_p.add_argument(
        "--manifest",
        required=True,
        metavar="FILE",
        help="JSON manifest listing staged clips (must contain 'accepted' field).",
    )
    extract_p.add_argument(
        "--landmarks-dir",
        default="data/landmarks",
        metavar="DIR",
        help="Root output directory for .npz skeleton files (default: data/landmarks).",
    )
    extract_p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List accepted clips without writing any files.",
    )
    extract_p.set_defaults(func=cmd_extract)

    # ------------------------------------------------------------------
    # review
    # ------------------------------------------------------------------
    review_p = subparsers.add_parser(
        "review",
        help="Interactively approve or reject extracted skeleton files.",
        description=(
            "Present each unapproved .npz file in data/landmarks/ for "
            "human review.  Approved files are copied to data/templates/; "
            "rejected files are moved to data/rejected_landmarks/."
        ),
    )
    review_p.add_argument(
        "--email",
        dest="reviewer_email",
        default="",
        metavar="EMAIL",
        help="Reviewer e-mail address recorded in the approval manifest.",
    )
    review_p.add_argument(
        "--mp4",
        dest="save_mp4",
        action="store_true",
        default=False,
        help="Save an MP4 preview instead of rendering in the terminal.",
    )
    review_p.set_defaults(func=cmd_review)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int | None:
    """Parse *argv* and dispatch to the appropriate sub-command handler.

    Returns the integer exit code for sub-commands that return one (e.g.
    ``extract``, ``review``).  Sub-commands such as ``fetch`` call
    ``sys.exit`` directly and therefore never return.

    When no sub-command is supplied the help text is printed and the process
    exits with code 0.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        # No sub-command supplied — print help and exit cleanly.
        parser.print_help()
        sys.exit(0)

    _configure_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    result = main()
    if result is not None:
        sys.exit(result)
