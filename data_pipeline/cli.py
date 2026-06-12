"""Command-line interface for the Kime data pipeline.

Usage
-----
List accepted clips without processing them::

    python -m data_pipeline.cli extract --dry-run

Run the extraction pipeline over all accepted staged clips::

    python -m data_pipeline.cli extract

Re-process clips that already have .npz output::

    python -m data_pipeline.cli extract --overwrite

Override default paths::

    python -m data_pipeline.cli extract \\
        --manifest data/staged/manifest.json \\
        --landmarks-dir data/landmarks \\
        --model-path models/pose_landmarker_lite.task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_pipeline.extractor import (
    DEFAULT_LANDMARKS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_STAGED_MANIFEST,
    extract_all,
    load_accepted_clips,
)


def _cmd_extract(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    landmarks_dir = Path(args.landmarks_dir)
    model_path = Path(args.model_path)

    if not manifest_path.exists():
        print(
            f"Error: manifest not found at '{manifest_path}'.\n"
            "Create data/staged/manifest.json listing your staged clips.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        accepted = load_accepted_clips(manifest_path)
        if not accepted:
            print("No accepted clips in manifest.")
        else:
            print(f"Accepted clips ({len(accepted)}):")
            for clip in accepted:
                print(
                    f"  [{clip['technique']}] {clip['clip_id']}"
                    f"  ← {clip['path']}"
                )
        return 0

    produced = extract_all(
        manifest_path=manifest_path,
        landmarks_dir=landmarks_dir,
        model_path=model_path,
        overwrite=args.overwrite,
    )
    print(f"\nDone. {len(produced)} .npz file(s) in '{landmarks_dir}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.cli",
        description="Kime data pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── extract sub-command ───────────────────────────────────────────────────
    ex = sub.add_parser(
        "extract",
        help="Extract skeleton landmarks from accepted staged clips.",
        description=(
            "For every accepted entry in the staged-clip manifest, run "
            "MediaPipe Pose frame-by-frame, apply Savitzky-Golay smoothing, "
            "normalise the skeleton (hip origin, unit torso height), and save "
            "a compressed .npz file under data/landmarks/<technique>/<clip_id>.npz."
        ),
    )
    ex.add_argument(
        "--manifest",
        default=str(DEFAULT_STAGED_MANIFEST),
        metavar="PATH",
        help=(
            "Path to the JSON clip manifest "
            f"(default: {DEFAULT_STAGED_MANIFEST})."
        ),
    )
    ex.add_argument(
        "--landmarks-dir",
        default=str(DEFAULT_LANDMARKS_DIR),
        metavar="DIR",
        help=(
            "Root directory for .npz output "
            f"(default: {DEFAULT_LANDMARKS_DIR})."
        ),
    )
    ex.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        metavar="PATH",
        help=(
            "Path to the MediaPipe pose landmarker .task bundle "
            f"(default: {DEFAULT_MODEL_PATH}, downloaded automatically)."
        ),
    )
    ex.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process clips that already have an output .npz file.",
    )
    ex.add_argument(
        "--dry-run",
        action="store_true",
        help="List accepted clips without running extraction.",
    )
    ex.set_defaults(func=_cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
