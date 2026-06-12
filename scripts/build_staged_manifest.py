"""Regenerate data/staged/manifest.json from clips that survived the filter.

``python -m data_pipeline.cli fetch`` downloads clips into
``data/staged/<technique>/`` and moves quality-filter rejects into
``data/staged/rejected/``, but nothing writes the manifest that the
``extract`` step consumes.  This script lists every surviving clip and marks
it ``accepted: true`` (acceptance here means "passed the automated
single-person filter" — human review happens later, at skeleton level, via
``python -m data_pipeline.cli review``).

Usage::

    python scripts/build_staged_manifest.py [--staging-dir data/staged]
"""

from __future__ import annotations

import argparse
import json
import pathlib

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
REJECTED_DIR_NAME = "rejected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=pathlib.Path,
                        default=pathlib.Path("data/staged"))
    args = parser.parse_args()

    staging: pathlib.Path = args.staging_dir
    entries = []
    for tech_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        if tech_dir.name == REJECTED_DIR_NAME:
            continue
        for clip in sorted(tech_dir.iterdir()):
            if clip.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append({
                    "technique": tech_dir.name,
                    "clip_id": clip.stem,
                    "path": str(clip.as_posix()),
                    "accepted": True,
                })

    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
    by_tech: dict[str, int] = {}
    for e in entries:
        by_tech[e["technique"]] = by_tech.get(e["technique"], 0) + 1
    print(f"Wrote {manifest_path} with {len(entries)} accepted clip(s):")
    for tech, count in sorted(by_tech.items()):
        print(f"  {tech}: {count}")


if __name__ == "__main__":
    main()
