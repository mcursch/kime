"""
Tests for backend.vision.segment.find_rep_window.

Coverage target: ≥95 % of segment.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.vision.segment import (
    SegmentationError,
    _joint_velocity,
    _LANDMARK_SETS,
    _DEFAULT_LANDMARKS,
    find_rep_window,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LANDMARKS = 33
N_DIMS = 3


def _make_poses(n_frames: int, velocity_profile: np.ndarray | None = None) -> np.ndarray:
    """Build a (n_frames, 33, 3) pose array.

    If *velocity_profile* (length == n_frames) is given the wrist landmarks
    move along the x-axis at the given per-frame speed; all other landmarks
    are static.  This makes the velocity of the wrist subset equal to
    *velocity_profile* (up to the prepend trick at frame 0).
    """
    poses = np.zeros((n_frames, N_LANDMARKS, N_DIMS))
    if velocity_profile is not None:
        assert len(velocity_profile) == n_frames
        # Integrate the velocity profile into wrist x-coordinates.
        # Wrists are landmarks 15 & 16.
        positions = np.cumsum(velocity_profile)
        for wrist_idx in (15, 16):
            poses[:, wrist_idx, 0] = positions
        # Also move ankles (27, 28) identically so the default set behaves.
        for ankle_idx in (27, 28):
            poses[:, ankle_idx, 0] = positions
    return poses


def _triangular_velocity(n_frames: int, peak_frame: int) -> np.ndarray:
    """Return a triangular velocity profile that peaks at *peak_frame*."""
    t = np.arange(n_frames, dtype=float)
    # Ramp up to peak, ramp down.
    profile = np.where(
        t <= peak_frame,
        t / peak_frame,
        (n_frames - 1 - t) / (n_frames - 1 - peak_frame),
    )
    return np.clip(profile, 0.0, 1.0)


# ---------------------------------------------------------------------------
# _joint_velocity unit tests
# ---------------------------------------------------------------------------


class TestJointVelocity:
    def test_output_length_matches_n_frames(self):
        n_frames = 20
        poses = _make_poses(n_frames)
        vel = _joint_velocity(poses, _DEFAULT_LANDMARKS)
        assert vel.shape == (n_frames,)

    def test_static_landmarks_give_zero_velocity(self):
        poses = _make_poses(30)  # all zeros → no movement
        vel = _joint_velocity(poses, _DEFAULT_LANDMARKS)
        np.testing.assert_array_equal(vel, 0.0)

    def test_moving_landmark_gives_positive_velocity(self):
        n_frames = 10
        profile = np.ones(n_frames) * 0.5
        poses = _make_poses(n_frames, profile)
        vel = _joint_velocity(poses, [15])  # left wrist only
        assert (vel > 0).all()

    def test_velocity_matches_expected_magnitude(self):
        """A wrist moving 1.0 unit/frame should give velocity ≈ 1.0."""
        n_frames = 5
        profile = np.ones(n_frames)
        poses = _make_poses(n_frames, profile)
        vel = _joint_velocity(poses, [15])
        # Frame 0 is copied from frame 1; frames 1-4 should be 1.0.
        np.testing.assert_allclose(vel[1:], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# find_rep_window – happy-path tests
# ---------------------------------------------------------------------------


class TestFindRepWindowHappyPath:
    def test_triangular_profile_extension_within_2_frames(self):
        """Extension frame must be within 2 frames of the true velocity peak."""
        n_frames = 60
        known_peak = 35
        profile = _triangular_velocity(n_frames, known_peak)
        poses = _make_poses(n_frames, profile)

        chamber, extension, retraction = find_rep_window(poses)

        assert abs(extension - known_peak) <= 2, (
            f"extension={extension} is more than 2 frames from known_peak={known_peak}"
        )

    def test_ordering_invariant(self):
        """chamber < extension < retraction must hold."""
        n_frames = 60
        profile = _triangular_velocity(n_frames, peak_frame=30)
        poses = _make_poses(n_frames, profile)

        chamber, extension, retraction = find_rep_window(poses)

        assert chamber < extension < retraction

    def test_ordering_invariant_early_peak(self):
        n_frames = 50
        profile = _triangular_velocity(n_frames, peak_frame=10)
        poses = _make_poses(n_frames, profile)
        chamber, extension, retraction = find_rep_window(poses)
        assert chamber < extension < retraction

    def test_ordering_invariant_late_peak(self):
        n_frames = 50
        profile = _triangular_velocity(n_frames, peak_frame=40)
        poses = _make_poses(n_frames, profile)
        chamber, extension, retraction = find_rep_window(poses)
        assert chamber < extension < retraction

    def test_straight_punch_movement_type(self):
        n_frames = 60
        profile = _triangular_velocity(n_frames, peak_frame=30)
        poses = _make_poses(n_frames, profile)
        chamber, extension, retraction = find_rep_window(poses, "straight_punch")
        assert chamber < extension < retraction

    def test_front_kick_movement_type(self):
        n_frames = 60
        # Build poses where ankles/knees move.
        profile = _triangular_velocity(n_frames, peak_frame=30)
        poses = np.zeros((n_frames, N_LANDMARKS, N_DIMS))
        positions = np.cumsum(profile)
        for idx in (25, 26, 27, 28):  # knees + ankles
            poses[:, idx, 0] = positions
        chamber, extension, retraction = find_rep_window(poses, "front_kick")
        assert chamber < extension < retraction

    def test_roundhouse_kick_movement_type(self):
        n_frames = 60
        profile = _triangular_velocity(n_frames, peak_frame=30)
        poses = np.zeros((n_frames, N_LANDMARKS, N_DIMS))
        positions = np.cumsum(profile)
        for idx in (25, 26, 27, 28):
            poses[:, idx, 0] = positions
        chamber, extension, retraction = find_rep_window(poses, "roundhouse_kick")
        assert chamber < extension < retraction

    def test_unknown_movement_type_uses_default(self):
        """Unrecognised movement_type should fall back to defaults without error."""
        n_frames = 60
        profile = _triangular_velocity(n_frames, peak_frame=30)
        poses = _make_poses(n_frames, profile)
        chamber, extension, retraction = find_rep_window(poses, "unknown_strike")
        assert chamber < extension < retraction

    def test_returns_integer_indices(self):
        n_frames = 40
        profile = _triangular_velocity(n_frames, peak_frame=20)
        poses = _make_poses(n_frames, profile)
        result = find_rep_window(poses)
        for idx in result:
            assert isinstance(idx, int)

    def test_numpy_array_input(self):
        """Accepts numpy arrays directly."""
        n_frames = 40
        profile = _triangular_velocity(n_frames, peak_frame=20)
        poses = _make_poses(n_frames, profile)
        chamber, extension, retraction = find_rep_window(np.asarray(poses))
        assert chamber < extension < retraction

    def test_list_input_is_accepted(self):
        """Accepts list-of-lists (coerced to ndarray internally)."""
        n_frames = 40
        profile = _triangular_velocity(n_frames, peak_frame=20)
        poses = _make_poses(n_frames, profile).tolist()
        chamber, extension, retraction = find_rep_window(poses)
        assert chamber < extension < retraction


# ---------------------------------------------------------------------------
# find_rep_window – error path tests
# ---------------------------------------------------------------------------


class TestFindRepWindowErrors:
    def test_flat_velocity_raises_segmentation_error(self):
        """A sequence with no velocity change must raise SegmentationError."""
        poses = _make_poses(40)  # all zeros → flat velocity
        with pytest.raises(SegmentationError, match="flat"):
            find_rep_window(poses)

    def test_monotonically_increasing_velocity_raises_segmentation_error(self):
        """Monotonically increasing velocity has non-zero range but no peaks."""
        n_frames = 40
        poses = np.zeros((n_frames, N_LANDMARKS, N_DIMS))
        # Accelerating motion: position is quadratic → velocity is linear (no peak).
        positions = np.arange(n_frames, dtype=float) ** 2
        for idx in (15, 16, 27, 28):
            poses[:, idx, 0] = positions
        with pytest.raises(SegmentationError, match="No velocity peak"):
            find_rep_window(poses)

    def test_constant_nonzero_velocity_raises_segmentation_error(self):
        """Constant (non-zero) velocity has no peaks and must raise SegmentationError."""
        # All wrists advance at a fixed rate → velocity is perfectly constant,
        # so find_peaks returns nothing and we cannot locate the rep window.
        n_frames = 40
        poses = np.zeros((n_frames, N_LANDMARKS, N_DIMS))
        positions = np.arange(n_frames, dtype=float)  # 1 unit/frame, constant speed
        for wrist_idx in (15, 16, 27, 28):
            poses[:, wrist_idx, 0] = positions
        with pytest.raises(SegmentationError):
            find_rep_window(poses)

    def test_too_few_frames_raises_segmentation_error(self):
        poses = _make_poses(3)
        with pytest.raises(SegmentationError, match="too short"):
            find_rep_window(poses)

    def test_wrong_shape_2d_raises_segmentation_error(self):
        poses = np.zeros((40, 33))
        with pytest.raises(SegmentationError, match="shape"):
            find_rep_window(poses)

    def test_wrong_landmark_count_raises_segmentation_error(self):
        poses = np.zeros((40, 17, 3))
        with pytest.raises(SegmentationError, match="shape"):
            find_rep_window(poses)

    def test_wrong_dimension_count_raises_segmentation_error(self):
        poses = np.zeros((40, 33, 2))
        with pytest.raises(SegmentationError, match="shape"):
            find_rep_window(poses)


# ---------------------------------------------------------------------------
# Landmark set configuration tests
# ---------------------------------------------------------------------------


class TestLandmarkSets:
    def test_straight_punch_uses_wrists_only(self):
        indices = _LANDMARK_SETS["straight_punch"]
        assert 15 in indices and 16 in indices
        assert 27 not in indices and 28 not in indices

    def test_kick_types_include_ankles(self):
        for technique in ("front_kick", "roundhouse_kick"):
            indices = _LANDMARK_SETS[technique]
            assert 27 in indices and 28 in indices

    def test_default_landmarks_include_wrists_and_ankles(self):
        for idx in (15, 16, 27, 28):
            assert idx in _DEFAULT_LANDMARKS
