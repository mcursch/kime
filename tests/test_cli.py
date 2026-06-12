"""Tests for data_pipeline.cli."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from data_pipeline.cli import _build_parser, main


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_fetch_parser_requires_sources() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch"])


def test_fetch_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["fetch", "--sources", "clips.yaml"])
    assert args.sources == "clips.yaml"
    assert args.staging_dir == "data/raw"
    assert args.max_duration == 60
    assert args.n_samples == 10
    assert abs(args.rejection_threshold - 0.5) < 1e-9


def test_fetch_parser_custom_values() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "fetch",
        "--sources", "my_clips.json",
        "--staging-dir", "/tmp/staging",
        "--max-duration", "30",
        "--n-samples", "5",
        "--rejection-threshold", "0.6",
    ])
    assert args.sources == "my_clips.json"
    assert args.staging_dir == "/tmp/staging"
    assert args.max_duration == 30
    assert args.n_samples == 5
    assert abs(args.rejection_threshold - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# cmd_fetch — integration-level (mocked internals)
# ---------------------------------------------------------------------------


def test_cmd_fetch_calls_fetch_and_filter(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "clips.yaml"
    src.write_text("front_kick:\n  - https://example.com/a\n")
    staging = tmp_path / "staging"

    fake_download = [staging / "front_kick" / "a.mp4"]
    fake_accepted = [staging / "front_kick" / "a.mp4"]
    fake_rejected: list[pathlib.Path] = []

    with (
        patch("data_pipeline.cli.fetch_all", return_value=fake_download) as mock_fetch,
        patch(
            "data_pipeline.cli.filter_all",
            return_value=(fake_accepted, fake_rejected),
        ) as mock_filter,
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["fetch", "--sources", str(src), "--staging-dir", str(staging)])

    assert exc_info.value.code == 0
    mock_fetch.assert_called_once()
    mock_filter.assert_called_once()


def test_cmd_fetch_no_downloads_exits_zero(tmp_path: pathlib.Path) -> None:
    """Even if nothing was downloaded, the command exits 0 (not a crash)."""
    src = tmp_path / "clips.yaml"
    src.write_text("front_kick:\n  - https://example.com/404\n")
    staging = tmp_path / "staging"

    with (
        patch("data_pipeline.cli.fetch_all", return_value=[]),
        patch("data_pipeline.cli.filter_all", return_value=([], [])),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["fetch", "--sources", str(src), "--staging-dir", str(staging)])

    assert exc_info.value.code == 0


def test_cmd_fetch_filter_receives_staging_dir(tmp_path: pathlib.Path) -> None:
    """filter_all is called with the same staging dir as fetch_all."""
    src = tmp_path / "clips.yaml"
    src.write_text("label:\n  - https://example.com/x\n")
    staging = tmp_path / "my_staging"

    with (
        patch("data_pipeline.cli.fetch_all", return_value=[staging / "label" / "x.mp4"]),
        patch("data_pipeline.cli.filter_all", return_value=([], [])) as mock_filter,
        pytest.raises(SystemExit),
    ):
        main(["fetch", "--sources", str(src), "--staging-dir", str(staging)])

    # filter_all is called with keyword arguments; staging_dir is first kwarg.
    kwargs = mock_filter.call_args.kwargs
    assert pathlib.Path(kwargs["staging_dir"]) == staging
