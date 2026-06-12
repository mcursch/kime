"""
End-to-end tests for backend.vision.pipeline.

Tests cover:
  - PreprocessResult is importable and has typed fields
  - preprocess runs without error on a synthetic sequence (round-trip test)
  - Fewer than 10 frames raises ValueError before segmentation is reached
  - The returned landmark array has the expected shape
  - The three frame indices are integers in ascending order within bounds
  - All movement_type variants accepted by find_rep_window work through the pipeline
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from backend.vision.pipeline import PreprocessResult, preprocess


# ---------------------------------------------------------------------------
# Synthetic landmark factory
# ---------------------------------------------------------------------------

def _make_standing_skeleton() -> np.ndarray:
    """Return a single-frame (33, 3) skeleton in a neutral standing pose.

    Only the landmarks needed by the pipeline modules are given realistic
    values; the rest are left at zero.

    MediaPipe indices used:
      11 – left shoulder   12 – right shoulder
      23 – left hip        24 – right hip
      15 – left wrist      16 – right wrist
      27 – left ankle      28 – right ankle
    """
    lm = np.zeros((33, 3), dtype=float)
    # Hips — straddling the origin slightly left/right
    lm[23] = [-0.10, 0.00, 0.00]   # left hip
    lm[24] = [ 0.10, 0.00, 0.00]   # right hip
    # Shoulders — above and slightly wider than hips
    lm[11] = [-0.20, 0.50, 0.00]   # left shoulder
    lm[12] = [ 0.20, 0.50, 0.00]   # right shoulder
    # Wrists at guard position
    lm[15] = [-0.30, 0.20, 0.10]   # left wrist
    lm[16] = [ 0.30, 0.20, 0.10]   # right wrist
    # Ankles below hips
    lm[27] = [-0.10, -1.00, 0.00]  # left ankle
    lm[28] = [ 0.10, -1.00, 0.00]  # right ankle
    return lm


def _make_front_kick_sequence(n_frames: int = 30) -> np.ndarray:
    """Synthesise an (N, 33, 3) sequence simulating a right-leg front kick.

    The right ankle (landmark 28) traces a smooth arc: it lifts during the
    chamber phase, reaches its highest point at the extension phase, then
    returns to the starting position.  All other landmarks stay at their
    standing-pose values to keep the sequence anatomically plausible.
    """
    base = _make_standing_skeleton()
    seq = np.tile(base[np.newaxis, :, :], (n_frames, 1, 1))  # (N, 33, 3)

    # Logistic (sigmoid) position profile: ankle extends smoothly from 0 to 1.
    # Its derivative — the velocity — is a bell-shaped curve peaking at the
    # sequence midpoint (an interior index), which scipy.signal.find_peaks can
    # detect reliably even after Savitzky-Golay smoothing.
    #
    # The previous sin(t) profile had velocity cos(t), which is monotonically
    # decreasing with its maximum at frame 0 (an endpoint index that find_peaks
    # never returns), causing SegmentationError in all round-trip tests.
    t = np.linspace(-4.0, 4.0, n_frames)
    raw = 1.0 / (1.0 + np.exp(-t))       # logistic: ~0 → ~1
    kick_height = raw - raw[0]            # start at 0
    if kick_height.max() > 0:
        kick_height /= kick_height.max() # peak at 1

    # Right ankle lifts up (positive Y) and forward (positive Z)
    seq[:, 28, 1] += kick_height * 0.8             # upward arc
    seq[:, 28, 2] += kick_height * 0.4             # slight forward reach

    return seq


def _make_punch_sequence(n_frames: int = 30) -> np.ndarray:
    """Synthesise an (N, 33, 3) sequence simulating a right straight punch."""
    base = _make_standing_skeleton()
    seq = np.tile(base[np.newaxis, :, :], (n_frames, 1, 1))

    # Same logistic profile as _make_front_kick_sequence; see that function
    # for a full explanation of why a sigmoid gives a detectable interior
    # velocity peak while the original sin(t) did not.
    t = np.linspace(-4.0, 4.0, n_frames)
    raw = 1.0 / (1.0 + np.exp(-t))
    punch_reach = raw - raw[0]
    if punch_reach.max() > 0:
        punch_reach /= punch_reach.max()

    # Right wrist extends forward (positive Z)
    seq[:, 16, 2] += punch_reach * 0.5

    return seq


# ---------------------------------------------------------------------------
# Import / dataclass tests
# ---------------------------------------------------------------------------

class TestPreprocessResultDataclass:
    def test_importable(self):
        """PreprocessResult is importable from backend.vision.pipeline."""
        from backend.vision.pipeline import PreprocessResult  # noqa: F401

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(PreprocessResult)

    def test_has_required_fields(self):
        fields = {f.name: f for f in dataclasses.fields(PreprocessResult)}
        assert "landmarks" in fields, "missing field: landmarks"
        assert "chamber" in fields, "missing field: chamber"
        assert "extension" in fields, "missing field: extension"
        assert "retraction" in fields, "missing field: retraction"

    def test_field_types(self):
        import typing

        fields = {f.name: f for f in dataclasses.fields(PreprocessResult)}
        # When ``from __future__ import annotations`` is active the annotation
        # is stored as a string; resolve it via get_type_hints so the test
        # handles both evaluation modes.
        hints = typing.get_type_hints(PreprocessResult)
        assert hints["landmarks"] is np.ndarray
        assert hints["chamber"] is int
        assert hints["extension"] is int
        assert hints["retraction"] is int


# ---------------------------------------------------------------------------
# End-to-end round-trip tests
# ---------------------------------------------------------------------------

class TestPreprocessRoundTrip:
    def test_front_kick_runs_without_error(self):
        """preprocess completes end-to-end on a synthetic front-kick sequence."""
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "front_kick")
        assert isinstance(result, PreprocessResult)

    def test_straight_punch_runs_without_error(self):
        seq = _make_punch_sequence(n_frames=30)
        result = preprocess(seq, "straight_punch")
        assert isinstance(result, PreprocessResult)

    def test_unknown_movement_type_runs_without_error(self):
        """Unknown movement_type falls back gracefully in the segment module."""
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "spinning_heel_kick")
        assert isinstance(result, PreprocessResult)

    def test_output_landmark_shape_preserved(self):
        n = 30
        seq = _make_front_kick_sequence(n_frames=n)
        result = preprocess(seq, "front_kick")
        assert result.landmarks.shape == (n, 33, 3)

    def test_output_landmarks_are_float(self):
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "front_kick")
        assert np.issubdtype(result.landmarks.dtype, np.floating)

    def test_frame_indices_are_integers(self):
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "front_kick")
        assert isinstance(result.chamber, int)
        assert isinstance(result.extension, int)
        assert isinstance(result.retraction, int)

    def test_frame_indices_in_ascending_order(self):
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "front_kick")
        assert result.chamber < result.extension < result.retraction

    def test_frame_indices_within_bounds(self):
        n = 30
        seq = _make_front_kick_sequence(n_frames=n)
        result = preprocess(seq, "front_kick")
        assert 0 <= result.chamber
        assert result.retraction < n

    def test_hip_is_near_origin_after_normalisation(self):
        """After normalisation, the mean hip midpoint should be near (0,0,0)."""
        seq = _make_front_kick_sequence(n_frames=30)
        result = preprocess(seq, "front_kick")
        hip_mid = (
            result.landmarks[:, 23, :] + result.landmarks[:, 24, :]
        ) / 2.0
        # Not exactly zero because SG smoothing shifts values slightly,
        # but the mean over the sequence should be small.
        assert np.abs(hip_mid.mean(axis=0)).max() < 0.5

    def test_longer_sequence_works(self):
        """Works on a longer sequence (simulating ~2 s at 30 fps)."""
        seq = _make_front_kick_sequence(n_frames=60)
        result = preprocess(seq, "front_kick")
        assert result.landmarks.shape == (60, 33, 3)

    def test_minimum_valid_frame_count(self):
        """Exactly 10 frames should not raise."""
        seq = _make_front_kick_sequence(n_frames=10)
        result = preprocess(seq, "front_kick")
        assert isinstance(result, PreprocessResult)


# ---------------------------------------------------------------------------
# ValueError for short sequences
# ---------------------------------------------------------------------------

class TestShortSequenceRaisesValueError:
    @pytest.mark.parametrize("n_frames", [1, 3, 5, 9])
    def test_raises_value_error(self, n_frames):
        """preprocess raises ValueError for sequences shorter than 10 frames."""
        seq = _make_front_kick_sequence(n_frames=n_frames)
        with pytest.raises(ValueError, match="10"):
            preprocess(seq, "front_kick")

    def test_error_raised_before_segmentation(self, monkeypatch):
        """The ValueError is raised before find_rep_window is called."""
        called = []

        import backend.vision.pipeline as _pipeline  # noqa: PLC0415

        original = _pipeline.find_rep_window

        def spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(_pipeline, "find_rep_window", spy)

        seq = _make_front_kick_sequence(n_frames=5)
        with pytest.raises(ValueError):
            preprocess(seq, "front_kick")

        assert not called, "find_rep_window was called despite too-few frames"


# ---------------------------------------------------------------------------
# Shape / type validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_ndim_raises_value_error(self):
        bad = np.zeros((30, 33))  # 2-D, missing the coordinate axis
        with pytest.raises(ValueError):
            preprocess(bad, "front_kick")

    def test_wrong_landmark_count_raises_value_error(self):
        bad = np.zeros((30, 17, 3))  # 17 landmarks instead of 33
        with pytest.raises(ValueError):
            preprocess(bad, "front_kick")
