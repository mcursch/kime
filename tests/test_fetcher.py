"""Tests for data_pipeline.fetcher."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from data_pipeline import fetcher


# ---------------------------------------------------------------------------
# _load_sources
# ---------------------------------------------------------------------------


def test_load_sources_yaml(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "clips.yaml"
    src.write_text(
        "front_kick:\n"
        "  - https://example.com/a\n"
        "  - https://example.com/b\n"
        "roundhouse_kick:\n"
        "  - https://example.com/c\n"
    )
    data = fetcher._load_sources(src)
    assert data == {
        "front_kick": ["https://example.com/a", "https://example.com/b"],
        "roundhouse_kick": ["https://example.com/c"],
    }


def test_load_sources_json(tmp_path: pathlib.Path) -> None:
    import json

    src = tmp_path / "clips.json"
    payload = {"straight_punch": ["https://example.com/x"]}
    src.write_text(json.dumps(payload))
    data = fetcher._load_sources(src)
    assert data == payload


def test_load_sources_invalid_type(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="mapping"):
        fetcher._load_sources(src)


# ---------------------------------------------------------------------------
# download_clip — happy path
# ---------------------------------------------------------------------------


def _make_fake_downloaded_file(tmp_dir: pathlib.Path, name: str = "vid123.mp4") -> pathlib.Path:
    """Write a tiny dummy file to simulate a yt-dlp download."""
    p = tmp_dir / name
    p.write_bytes(b"\x00" * 16)
    return p


def test_download_clip_success(tmp_path: pathlib.Path) -> None:
    staging = tmp_path / "staging"

    fake_info = {"id": "vid123", "duration": 30}

    # Patch TemporaryDirectory to use a persistent path we control so we can
    # pre-populate it before yt-dlp would normally run.
    with tempfile.TemporaryDirectory() as real_tmp:
        real_tmp_path = pathlib.Path(real_tmp)
        _make_fake_downloaded_file(real_tmp_path)

        with (
            patch("data_pipeline.fetcher.tempfile.TemporaryDirectory") as mock_td,
            patch("data_pipeline.fetcher.yt_dlp.YoutubeDL") as mock_ydl_cls,
            patch("data_pipeline.fetcher.ffmpeg.input") as mock_ff_input,
        ):
            # TemporaryDirectory context manager yields real_tmp so our file exists.
            mock_td.return_value.__enter__.return_value = real_tmp
            mock_td.return_value.__exit__.return_value = False

            # yt-dlp context manager returns fake info dict.
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.extract_info.return_value = fake_info
            mock_ydl_cls.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_cls.return_value.__exit__.return_value = False

            # ffmpeg chain: .input().output().overwrite_output().run()
            mock_ff_chain = MagicMock()
            mock_ff_input.return_value = mock_ff_chain
            mock_ff_chain.output.return_value = mock_ff_chain
            mock_ff_chain.overwrite_output.return_value = mock_ff_chain
            mock_ff_chain.run.return_value = None

            result = fetcher.download_clip(
                url="https://example.com/watch?v=vid123",
                label="front_kick",
                staging_dir=staging,
                max_duration=60,
            )

    assert result is not None
    assert result == staging / "front_kick" / "vid123.mp4"


# ---------------------------------------------------------------------------
# download_clip — 404 / DownloadError should return None, not raise
# ---------------------------------------------------------------------------


def test_download_clip_404_skipped(tmp_path: pathlib.Path) -> None:
    import yt_dlp

    staging = tmp_path / "staging"

    with (
        patch("data_pipeline.fetcher.tempfile.TemporaryDirectory") as mock_td,
        patch("data_pipeline.fetcher.yt_dlp.YoutubeDL") as mock_ydl_cls,
    ):
        mock_td.return_value.__enter__.return_value = str(tmp_path / "empty_tmp")
        mock_td.return_value.__exit__.return_value = False
        (tmp_path / "empty_tmp").mkdir()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "HTTP Error 404: Not Found"
        )
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_cls.return_value.__exit__.return_value = False

        result = fetcher.download_clip(
            url="https://example.com/watch?v=missing",
            label="front_kick",
            staging_dir=staging,
            max_duration=60,
        )

    assert result is None


# ---------------------------------------------------------------------------
# fetch_all — end-to-end with mocked download_clip
# ---------------------------------------------------------------------------


def test_fetch_all_calls_download_for_each_url(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "clips.yaml"
    src.write_text(
        "front_kick:\n"
        "  - https://example.com/a\n"
        "  - https://example.com/b\n"
    )
    staging = tmp_path / "staging"

    fake_paths = [staging / "front_kick" / "a.mp4", staging / "front_kick" / "b.mp4"]

    with patch("data_pipeline.fetcher.download_clip", side_effect=fake_paths) as mock_dl:
        results = fetcher.fetch_all(src, staging, max_duration=30)

    assert mock_dl.call_count == 2
    assert results == fake_paths


def test_fetch_all_handles_partial_failures(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "clips.yaml"
    src.write_text(
        "roundhouse_kick:\n"
        "  - https://example.com/ok\n"
        "  - https://example.com/fail\n"
    )
    staging = tmp_path / "staging"
    good = staging / "roundhouse_kick" / "ok.mp4"

    with patch("data_pipeline.fetcher.download_clip", side_effect=[good, None]):
        results = fetcher.fetch_all(src, staging)

    # Only the successful download appears in the result list.
    assert results == [good]
