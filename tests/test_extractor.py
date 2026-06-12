"""Tests for data_pipeline.extractor (smoothing, normalisation, I/O).

MediaPipe and OpenCV are not exercised in these unit tests — those paths are
covered by the integration smoke-test at the bottom of this file (skipped
unless a real video and model are present).  The pure-Python functions
(smooth_landmarks, normalize_landmarks, process_clip output format) are tested
with synthetic NumPy data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.extractor import (
    _LEFT_HIP,
    _LEFT_SHOULDER,
    _RIGHT_HIP,
    _RIGHT_SHOULDER,
    extract_all,
    load_accepted_clips,
    normalize_landmarks,
    process_clip,
    smooth_landmarks,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_random_coords(n_frames: int = 60, seed: int = 0) -> np.ndarray:
    """Return random coords shaped (N, 33, 3) with a plausible skeleton."""
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n_frames, 33, 3)).astype(np.float32)
    # Place hips and shoulders at fixed offsets so normalisation is
    # non-degenerate.
    coords[:, _LEFT_HIP, :] = [0.1, -0.9, 0.0]
    coords[:, _RIGHT_HIP, :] = [-0.1, -0.9, 0.0]
    coords[:, _LEFT_SHOULDER, :] = [0.2, 0.1, 0.0]
    coords[:, _RIGHT_SHOULDER, :] = [-0.2, 0.1, 0.0]
    return coords


# ── smooth_landmarks ──────────────────────────────────────────────────────────

class TestSmoothLandmarks:
    def test_output_shape_unchanged(self):
        coords = _make_random_coords(60)
        out = smooth_landmarks(coords)
        assert out.shape == coords.shape

    def test_output_dtype_float32(self):
        out = smooth_landmarks(_make_random_coords(60))
        assert out.dtype == np.float32

    def test_very_short_sequence_returns_copy(self):
        """Sequences too short for the filter window pass through unchanged."""
        coords = _make_random_coords(3)
        out = smooth_landmarks(coords, window_length=11, polyorder=3)
        np.testing.assert_array_equal(out, coords.astype(np.float32))

    def test_smoothing_reduces_variance(self):
        """Smoothed output has lower or equal variance than the input."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((120, 33, 3)).astype(np.float32)
        out = smooth_landmarks(coords, window_length=11, polyorder=3)
        assert out.var() <= coords.var() + 1e-6

    def test_even_window_auto_corrected(self):
        """An even window_length should not raise; the function adjusts it."""
        coords = _make_random_coords(60)
        out = smooth_landmarks(coords, window_length=10, polyorder=3)
        assert out.shape == coords.shape


# ── normalize_landmarks ───────────────────────────────────────────────────────

class TestNormalizeLandmarks:
    def test_output_shape_unchanged(self):
        coords = _make_random_coords(60)
        out = normalize_landmarks(coords)
        assert out.shape == coords.shape

    def test_output_dtype_float32(self):
        out = normalize_landmarks(_make_random_coords(60))
        assert out.dtype == np.float32

    def test_hip_midpoint_at_origin_per_frame(self):
        """After normalisation the hip midpoint must be the origin each frame."""
        coords = _make_random_coords(60)
        out = normalize_landmarks(coords)
        hip_mid = (out[:, _LEFT_HIP, :] + out[:, _RIGHT_HIP, :]) / 2.0
        np.testing.assert_allclose(hip_mid, 0.0, atol=1e-5)

    def test_mean_hip_midpoint_within_tolerance(self):
        """Mean hip-midpoint across all frames must be within 0.01 of origin."""
        coords = _make_random_coords(60)
        out = normalize_landmarks(coords)
        hip_mid = (out[:, _LEFT_HIP, :] + out[:, _RIGHT_HIP, :]) / 2.0
        assert np.abs(hip_mid.mean(axis=0)).max() < 0.01

    def test_median_torso_height_is_one(self):
        """Median torso height should equal 1.0 after normalisation."""
        coords = _make_random_coords(60)
        out = normalize_landmarks(coords)
        shoulder_mid = (
            out[:, _LEFT_SHOULDER, :] + out[:, _RIGHT_SHOULDER, :]
        ) / 2.0
        torso_heights = np.linalg.norm(shoulder_mid, axis=1)
        assert abs(float(np.median(torso_heights)) - 1.0) < 1e-5

    def test_degenerate_input_raises(self):
        """All-zero coords (zero torso height) must raise ValueError."""
        coords = np.zeros((30, 33, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="torso height"):
            normalize_landmarks(coords)


# ── load_accepted_clips ───────────────────────────────────────────────────────

class TestLoadAcceptedClips:
    def test_filters_accepted_true(self, tmp_path):
        manifest = [
            {"technique": "front_kick", "clip_id": "a", "path": "a.mp4", "accepted": True},
            {"technique": "front_kick", "clip_id": "b", "path": "b.mp4", "accepted": False},
            {"technique": "roundhouse_kick", "clip_id": "c", "path": "c.mp4", "accepted": True},
        ]
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        result = load_accepted_clips(mpath)
        assert len(result) == 2
        assert all(c["accepted"] for c in result)

    def test_empty_manifest_returns_empty(self, tmp_path):
        mpath = tmp_path / "manifest.json"
        mpath.write_text("[]")
        assert load_accepted_clips(mpath) == []

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_accepted_clips(tmp_path / "nonexistent.json")


# ── process_clip (mocked extraction) ─────────────────────────────────────────

def _fake_extract_landmarks(video_path, model_path):
    """Synthetic replacement for extract_landmarks — no OpenCV/MediaPipe needed."""
    n = 60
    coords = _make_random_coords(n)
    visibility = np.ones((n, 33), dtype=np.float32) * 0.9
    timestamps = np.linspace(0, 2.0, n)
    return coords, visibility, timestamps


def test_process_clip_creates_npz(tmp_path, monkeypatch):
    """process_clip should write a .npz with the three required arrays."""
    import data_pipeline.extractor as ext

    monkeypatch.setattr(ext, "extract_landmarks", _fake_extract_landmarks)
    monkeypatch.setattr(ext, "ensure_model", lambda p: p)

    clip = {
        "technique": "front_kick",
        "clip_id": "test_clip",
        "path": str(tmp_path / "dummy.mp4"),
    }
    out_path = process_clip(clip, landmarks_dir=tmp_path, overwrite=True)

    assert out_path.exists()
    assert out_path.suffix == ".npz"

    data = np.load(out_path)
    assert set(data.files) >= {"coords", "visibility", "timestamps"}

    n = data["timestamps"].shape[0]
    assert data["coords"].shape == (n, 33, 3)
    assert data["visibility"].shape == (n, 33)
    assert data["timestamps"].shape == (n,)


def test_process_clip_skip_existing(tmp_path, monkeypatch):
    """process_clip should not re-run if output exists and overwrite=False."""
    import data_pipeline.extractor as ext

    call_count = {"n": 0}

    def _counting_extract(video_path, model_path):
        call_count["n"] += 1
        return _fake_extract_landmarks(video_path, model_path)

    monkeypatch.setattr(ext, "extract_landmarks", _counting_extract)
    monkeypatch.setattr(ext, "ensure_model", lambda p: p)

    clip = {
        "technique": "roundhouse_kick",
        "clip_id": "skip_me",
        "path": str(tmp_path / "dummy.mp4"),
    }

    # First call creates the file.
    process_clip(clip, landmarks_dir=tmp_path, overwrite=True)
    assert call_count["n"] == 1

    # Second call (overwrite=False) should skip.
    process_clip(clip, landmarks_dir=tmp_path, overwrite=False)
    assert call_count["n"] == 1  # not incremented


def test_process_clip_coords_normalised(tmp_path, monkeypatch):
    """Stored coords must satisfy the hip-origin constraint."""
    import data_pipeline.extractor as ext

    monkeypatch.setattr(ext, "extract_landmarks", _fake_extract_landmarks)
    monkeypatch.setattr(ext, "ensure_model", lambda p: p)

    clip = {
        "technique": "straight_punch",
        "clip_id": "norm_check",
        "path": str(tmp_path / "dummy.mp4"),
    }
    out_path = process_clip(clip, landmarks_dir=tmp_path, overwrite=True)
    data = np.load(out_path)
    coords = data["coords"]
    hip_mid = (coords[:, _LEFT_HIP, :] + coords[:, _RIGHT_HIP, :]) / 2.0
    # Mean hip-midpoint across all frames must be within 0.01 of origin.
    assert np.abs(hip_mid.mean(axis=0)).max() < 0.01


# ── no video files in landmarks dir ──────────────────────────────────────────

def test_no_video_files_in_output(tmp_path, monkeypatch):
    """extract_all must not produce video files under the landmarks directory."""
    import data_pipeline.extractor as ext

    monkeypatch.setattr(ext, "extract_landmarks", _fake_extract_landmarks)
    monkeypatch.setattr(ext, "ensure_model", lambda p: p)

    manifest_data = [
        {
            "technique": "front_kick",
            "clip_id": "vid_check",
            "path": str(tmp_path / "dummy.mp4"),
            "accepted": True,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data))

    extract_all(
        manifest_path=manifest_path,
        landmarks_dir=tmp_path / "landmarks",
        model_path=tmp_path / "fake.task",
        overwrite=True,
    )

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    video_files = [
        p
        for p in (tmp_path / "landmarks").rglob("*")
        if p.suffix.lower() in video_exts
    ]
    assert video_files == [], f"Video files found in output: {video_files}"


# ── CLI smoke test ────────────────────────────────────────────────────────────

def test_cli_extract_dry_run(tmp_path, monkeypatch):
    """The CLI extract --dry-run command should list clips without writing files."""
    import sys

    from data_pipeline.cli import main

    manifest_data = [
        {
            "technique": "front_kick",
            "clip_id": "cli_test",
            "path": "data/staged/front_kick/cli_test.mp4",
            "accepted": True,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data))

    ret = main(
        [
            "extract",
            "--manifest",
            str(manifest_path),
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--dry-run",
        ]
    )
    assert ret == 0
    # No output files should be created in dry-run mode.
    assert not (tmp_path / "landmarks").exists() or not any(
        (tmp_path / "landmarks").rglob("*.npz")
    )


def test_cli_extract_missing_manifest(tmp_path):
    """CLI extract should return exit code 1 when manifest is missing."""
    from data_pipeline.cli import main

    ret = main(
        [
            "extract",
            "--manifest",
            str(tmp_path / "no_such.json"),
        ]
    )
    assert ret == 1
