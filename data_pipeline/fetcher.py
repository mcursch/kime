"""
Fetcher — download clips listed in a YAML/JSON sources file and trim them.

Sources file format (YAML or JSON):
    front_kick:
      - https://www.youtube.com/watch?v=XXXXXXX
      - https://www.youtube.com/watch?v=YYYYYYY
    roundhouse_kick:
      - https://www.youtube.com/watch?v=ZZZZZZZ

Each URL is downloaded into ``<staging_dir>/<label>/`` and trimmed to
``max_duration`` seconds so downstream pose work stays manageable.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import shutil
import tempfile
from typing import Any

import ffmpeg  # ffmpeg-python
import yt_dlp
import yaml

logger = logging.getLogger(__name__)

# yt-dlp error types we treat as "skip this URL" rather than crash.
_SKIP_ERRORS = (
    yt_dlp.utils.DownloadError,
    yt_dlp.utils.ExtractorError,
)


def _load_sources(sources_file: str | pathlib.Path) -> dict[str, list[str]]:
    """Return ``{label: [url, ...]}`` from a YAML or JSON file."""
    path = pathlib.Path(sources_file)
    with path.open() as fh:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Sources file must be a mapping of label→[url …], got {type(data)}"
        )
    return {str(label): list(urls) for label, urls in data.items()}


def _safe_filename(url: str) -> str:
    """Derive a filesystem-safe base name from a URL (no extension)."""
    # Keep only alphanumerics, hyphens, underscores, and dots.
    slug = re.sub(r"[^\w\-.]", "_", url)
    return slug[:120]  # cap length


def download_clip(
    url: str,
    label: str,
    staging_dir: pathlib.Path,
    max_duration: int = 60,
) -> pathlib.Path | None:
    """Download *url* into ``staging_dir/label/`` and trim to *max_duration* seconds.

    Returns the path to the trimmed file, or ``None`` if the download fails
    (e.g. 404, private video, geo-block).  Failures are logged as warnings so
    the caller can keep processing remaining URLs.

    Parameters
    ----------
    url:
        Publicly accessible video URL understood by yt-dlp.
    label:
        Technique label (e.g. ``"front_kick"``).  Used as a sub-directory.
    staging_dir:
        Root directory for downloaded clips.
    max_duration:
        Maximum clip length in seconds.  The clip is hard-trimmed with ffmpeg
        when it exceeds this value; shorter clips are left as-is.
    """
    label_dir = staging_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kime_dl_") as tmp:
        tmp_path = pathlib.Path(tmp)
        out_template = str(tmp_path / "%(id)s.%(ext)s")

        ydl_opts: dict[str, Any] = {
            "outtmpl": out_template,
            # Prefer a single mp4/mkv file; fall back to best available.
            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/mp4/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            # Abort quickly on HTTP errors so we don't wait forever.
            "socket_timeout": 30,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except _SKIP_ERRORS as exc:
            logger.warning("Skipping %s (%s: %s)", url, type(exc).__name__, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error downloading %s: %s", url, exc)
            return None

        # Locate the downloaded file inside tmp.
        downloaded = list(tmp_path.glob("*"))
        if not downloaded:
            logger.warning("yt-dlp reported success but no file found for %s", url)
            return None
        src = downloaded[0]

        video_id = info.get("id", _safe_filename(url)) if info else _safe_filename(url)
        dest = label_dir / f"{video_id}.mp4"

        # Trim or copy to final location.
        duration: float | None = info.get("duration") if info else None
        try:
            if duration is not None and duration <= max_duration:
                # No trim needed — just copy/re-mux to ensure mp4 container.
                (
                    ffmpeg
                    .input(str(src))
                    .output(str(dest), c="copy", movflags="faststart")
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # Hard-trim to max_duration seconds.
                (
                    ffmpeg
                    .input(str(src), t=max_duration)
                    .output(str(dest), c="copy", movflags="faststart")
                    .overwrite_output()
                    .run(quiet=True)
                )
        except ffmpeg.Error as exc:
            logger.warning(
                "ffmpeg failed for %s: %s", url, exc.stderr.decode(errors="replace")
            )
            # Fall back to plain file copy so we at least have the raw file.
            shutil.copy2(src, dest)

    logger.info("Downloaded: %s → %s", url, dest)
    return dest


def fetch_all(
    sources_file: str | pathlib.Path,
    staging_dir: str | pathlib.Path,
    max_duration: int = 60,
) -> list[pathlib.Path]:
    """Download every clip listed in *sources_file*.

    Parameters
    ----------
    sources_file:
        Path to a YAML or JSON file mapping technique labels to lists of URLs.
    staging_dir:
        Root directory where clips are saved.
    max_duration:
        Maximum clip length in seconds (clips are trimmed to this length).

    Returns
    -------
    list[pathlib.Path]
        Paths of successfully downloaded clips.
    """
    staging = pathlib.Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(sources_file)
    succeeded: list[pathlib.Path] = []

    for label, urls in sources.items():
        logger.info("Fetching %d clip(s) for label '%s'", len(urls), label)
        for url in urls:
            result = download_clip(url, label, staging, max_duration=max_duration)
            if result is not None:
                succeeded.append(result)

    logger.info(
        "Fetch complete: %d clip(s) downloaded across %d label(s).",
        len(succeeded),
        len(sources),
    )
    return succeeded
