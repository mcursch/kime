"""Tests for data_pipeline.filter.

All MediaPipe / OpenCV interactions are mocked so the tests run without a
GPU or a downloaded model file.
"""

from __future__ import annotations

import logging
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_pipeline import filter as dp_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_frame(h: int = 480, w: int = 640) -> "np.ndarray":
    """Return a deterministic BGR frame for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _patch_sample_frames(frames):
    """Patch _sample_frames to return a fixed list of (idx, frame) tuples."""
    return patch(
        "data_pipeline.filter._sample_frames",
        return_value=[(i, f) for i, f in enumerate(frames)],
    )


def _patch_count_poses(results: list[int]):
    """Patch _count_poses to return values from *results* in order."""
    return patch("data_pipeline.filter._count_poses", side_effect=results)


def _patch_resolve_model(fake_path: pathlib.Path):
    """Patch _resolve_model to skip network access."""
    return patch("data_pipeline.filter._resolve_model", return_value=fake_path)


def _patch_landmarker():
    """Patch PoseLandmarker.create_from_options to avoid loading a real model."""
    mock_lm = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_lm)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return patch(
        "data_pipeline.filter.mp_vision.PoseLandmarker.create_from_options",
        return_value=mock_ctx,
    )


# ---------------------------------------------------------------------------
# filter_clip — acceptance
# ---------------------------------------------------------------------------


def test_filter_clip_accepts_single_person(tmp_path: pathlib.Path) -> None:
    """All frames have exactly one pose → clip accepted."""
    clip = tmp_path / "front_kick" / "good.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(5)]
    pose_counts = [1] * 5  # exactly one person in every frame

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
    ):
        accepted = dp_filter.filter_clip(clip, tmp_path)

    assert accepted is True
    assert clip.exists(), "Accepted clip must stay in place"


# ---------------------------------------------------------------------------
# filter_clip — no-pose rejection
# ---------------------------------------------------------------------------


def test_filter_clip_rejects_no_pose(tmp_path: pathlib.Path) -> None:
    """Majority of frames have no pose → rejected as 'no pose detected'."""
    clip = tmp_path / "front_kick" / "empty.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(6)]
    # 4 of 6 frames return 0 poses (> 50%)
    pose_counts = [0, 0, 0, 0, 1, 1]

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
    ):
        accepted = dp_filter.filter_clip(clip, tmp_path)

    assert accepted is False

    rejected_path = (
        tmp_path / dp_filter.REJECTION_DIR_NAME / "front_kick" / "empty.mp4"
    )
    assert rejected_path.exists(), "Rejected clip must be moved to rejected/ dir"
    assert not clip.exists(), "Original path must be absent after rejection"


def test_filter_clip_no_pose_logged(tmp_path: pathlib.Path, caplog) -> None:
    """'no pose detected' appears in the log for no-pose rejection."""
    clip = tmp_path / "label" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(4)]
    pose_counts = [0] * 4

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
        caplog.at_level(logging.WARNING, logger="data_pipeline.filter"),
    ):
        dp_filter.filter_clip(clip, tmp_path)

    assert any("no pose detected" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# filter_clip — multi-person rejection
# ---------------------------------------------------------------------------


def test_filter_clip_rejects_multi_person(tmp_path: pathlib.Path) -> None:
    """Majority of frames have >1 pose → rejected as 'multi-person detected'."""
    clip = tmp_path / "front_kick" / "crowd.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(4)]
    # 3 of 4 frames return 2 poses (> 50%)
    pose_counts = [2, 2, 2, 1]

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
    ):
        accepted = dp_filter.filter_clip(clip, tmp_path)

    assert accepted is False

    rejected_path = (
        tmp_path / dp_filter.REJECTION_DIR_NAME / "front_kick" / "crowd.mp4"
    )
    assert rejected_path.exists()


def test_filter_clip_multi_person_logged(tmp_path: pathlib.Path, caplog) -> None:
    """'multi-person detected' appears in the log for multi-person rejection."""
    clip = tmp_path / "label" / "crowd.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(4)]
    pose_counts = [3] * 4  # always 3 people

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
        caplog.at_level(logging.WARNING, logger="data_pipeline.filter"),
    ):
        dp_filter.filter_clip(clip, tmp_path)

    assert any("multi-person detected" in r.message for r in caplog.records)


def test_filter_clip_multi_person_takes_priority_over_no_pose(
    tmp_path: pathlib.Path, caplog
) -> None:
    """When both conditions exceed the threshold, multi-person is reported."""
    clip = tmp_path / "label" / "mixed.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    frames = [_fake_frame() for _ in range(4)]
    # 2 frames: no pose; 2 frames: 2 poses → both at 50% threshold
    pose_counts = [0, 0, 2, 2]

    with (
        _patch_sample_frames(frames),
        _patch_count_poses(pose_counts),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
        caplog.at_level(logging.WARNING, logger="data_pipeline.filter"),
    ):
        accepted = dp_filter.filter_clip(clip, tmp_path, rejection_threshold=0.5)

    assert accepted is False
    assert any("multi-person detected" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# filter_clip — no frames sampled (unreadable video)
# ---------------------------------------------------------------------------


def test_filter_clip_rejects_unreadable_video(tmp_path: pathlib.Path) -> None:
    """A video with no readable frames is rejected."""
    clip = tmp_path / "label" / "broken.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00")

    with (
        patch("data_pipeline.filter._sample_frames", return_value=[]),
        _patch_resolve_model(tmp_path / "fake.task"),
        _patch_landmarker(),
    ):
        accepted = dp_filter.filter_clip(clip, tmp_path)

    assert accepted is False


# ---------------------------------------------------------------------------
# filter_all
# ---------------------------------------------------------------------------


def test_filter_all_skips_already_rejected(tmp_path: pathlib.Path) -> None:
    """Clips already in rejected/ are not re-processed."""
    (tmp_path / "front_kick").mkdir()
    good = tmp_path / "front_kick" / "good.mp4"
    good.write_bytes(b"\x00")

    already_rejected = tmp_path / "rejected" / "front_kick" / "bad.mp4"
    already_rejected.parent.mkdir(parents=True)
    already_rejected.write_bytes(b"\x00")

    with patch("data_pipeline.filter.filter_clip", return_value=True) as mock_fc:
        accepted, rejected = dp_filter.filter_all(tmp_path)

    # filter_clip must only be called for good.mp4 (bad.mp4 is already in rejected/).
    assert mock_fc.call_count == 1
    called_path = mock_fc.call_args[0][0]
    assert called_path.name == "good.mp4"
    assert len(accepted) == 1
    assert len(rejected) == 0


def test_filter_all_counts_accepted_and_rejected(tmp_path: pathlib.Path) -> None:
    """filter_all aggregates results across multiple clips."""
    (tmp_path / "label").mkdir()
    for name in ["a.mp4", "b.mp4", "c.mp4"]:
        (tmp_path / "label" / name).write_bytes(b"\x00")

    # a → accepted, b → rejected, c → accepted
    with patch("data_pipeline.filter.filter_clip", side_effect=[True, False, True]):
        accepted, rejected = dp_filter.filter_all(tmp_path)

    assert len(accepted) == 2
    assert len(rejected) == 1


def test_filter_all_passes_model_path(tmp_path: pathlib.Path) -> None:
    """The model_path argument is forwarded to each filter_clip call."""
    (tmp_path / "label").mkdir()
    clip = tmp_path / "label" / "x.mp4"
    clip.write_bytes(b"\x00")

    fake_model = pathlib.Path("/fake/model.task")

    with patch("data_pipeline.filter.filter_clip", return_value=True) as mock_fc:
        dp_filter.filter_all(tmp_path, model_path=fake_model)

    _, kwargs = mock_fc.call_args
    assert kwargs.get("model_path") == fake_model
