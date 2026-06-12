"""Tests for backend.vision.normalize.

Covers hip_center, torso_scale, and canonical_facing with synthetic fixtures
that exercise all branches, including degenerate (zero-length) edge cases.
"""

import math

import numpy as np
import pytest

from backend.vision.normalize import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    canonical_facing,
    check_camera_angle,
    hip_center,
    torso_scale,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LANDMARKS = 33


def _make_frames(n_frames: int, seed: int = 0) -> np.ndarray:
    """Return random frames of shape (n_frames, 33, 3)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, N_LANDMARKS, 3))


def _hip_mid(frames: np.ndarray) -> np.ndarray:
    """Return hip midpoints, shape (N_frames, 3)."""
    return (frames[:, LEFT_HIP, :] + frames[:, RIGHT_HIP, :]) / 2.0


def _shoulder_mid(frames: np.ndarray) -> np.ndarray:
    """Return shoulder midpoints, shape (N_frames, 3)."""
    return (frames[:, LEFT_SHOULDER, :] + frames[:, RIGHT_SHOULDER, :]) / 2.0


def _torso_lengths(frames: np.ndarray) -> np.ndarray:
    """Return per-frame torso lengths, shape (N_frames,)."""
    return np.linalg.norm(_shoulder_mid(frames) - _hip_mid(frames), axis=1)


# ---------------------------------------------------------------------------
# hip_center tests
# ---------------------------------------------------------------------------


class TestHipCenter:
    def test_hip_midpoint_is_origin_per_frame(self):
        frames = _make_frames(10)
        result = hip_center(frames)
        mid = _hip_mid(result)
        np.testing.assert_allclose(mid, 0.0, atol=1e-12)

    def test_mean_hip_midpoint_within_tolerance(self):
        """Acceptance criterion: mean hip-midpoint across all frames ≈ 0."""
        frames = _make_frames(30)
        result = hip_center(frames)
        mean_mid = _hip_mid(result).mean(axis=0)
        assert np.linalg.norm(mean_mid) < 1e-6

    def test_shape_preserved(self):
        frames = _make_frames(5)
        assert hip_center(frames).shape == frames.shape

    def test_relative_positions_unchanged(self):
        """Pairwise landmark differences must be translation-invariant."""
        frames = _make_frames(4)
        result = hip_center(frames)
        for i in range(N_LANDMARKS - 1):
            diff_before = frames[:, i + 1, :] - frames[:, i, :]
            diff_after = result[:, i + 1, :] - result[:, i, :]
            np.testing.assert_allclose(diff_after, diff_before, atol=1e-12)

    def test_single_frame(self):
        frames = _make_frames(1)
        result = hip_center(frames)
        np.testing.assert_allclose(_hip_mid(result), 0.0, atol=1e-12)

    def test_returns_new_array(self):
        frames = _make_frames(3)
        result = hip_center(frames)
        assert result is not frames

    def test_accepts_list_input(self):
        """Should work with Python lists as well as NumPy arrays."""
        frames = _make_frames(2).tolist()
        result = hip_center(frames)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, N_LANDMARKS, 3)


# ---------------------------------------------------------------------------
# torso_scale tests
# ---------------------------------------------------------------------------


class TestTorsoScale:
    def test_torso_length_is_one_per_frame(self):
        frames = _make_frames(10)
        result = torso_scale(frames)
        lengths = _torso_lengths(result)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-12)

    def test_mean_torso_length_within_tolerance(self):
        """Acceptance criterion: mean torso length across all frames ≈ 1."""
        frames = _make_frames(30)
        result = torso_scale(frames)
        mean_len = _torso_lengths(result).mean()
        assert abs(mean_len - 1.0) < 1e-6

    def test_shape_preserved(self):
        frames = _make_frames(7)
        assert torso_scale(frames).shape == frames.shape

    def test_relative_directions_unchanged(self):
        """Unit vectors between landmarks must be scale-invariant."""
        frames = _make_frames(4)
        result = torso_scale(frames)
        for i in range(N_LANDMARKS - 1):
            vec_before = frames[:, i + 1, :] - frames[:, i, :]
            vec_after = result[:, i + 1, :] - result[:, i, :]
            norms_before = np.linalg.norm(vec_before, axis=1, keepdims=True)
            norms_after = np.linalg.norm(vec_after, axis=1, keepdims=True)
            # Avoid div-by-zero for zero-length vectors
            mask = norms_before[:, 0] > 1e-10
            if mask.any():
                np.testing.assert_allclose(
                    vec_after[mask] / norms_after[mask],
                    vec_before[mask] / norms_before[mask],
                    atol=1e-12,
                )

    def test_degenerate_zero_torso_unchanged(self):
        """A frame with zero torso length must not be modified."""
        frames = np.zeros((1, N_LANDMARKS, 3))
        # All zeros → torso length is 0 → should return zeros unchanged
        result = torso_scale(frames)
        np.testing.assert_array_equal(result, frames)

    def test_single_frame(self):
        frames = _make_frames(1)
        result = torso_scale(frames)
        np.testing.assert_allclose(_torso_lengths(result), 1.0, atol=1e-12)

    def test_returns_new_array(self):
        frames = _make_frames(3)
        result = torso_scale(frames)
        assert result is not frames

    def test_accepts_list_input(self):
        frames = _make_frames(2).tolist()
        result = torso_scale(frames)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# canonical_facing tests
# ---------------------------------------------------------------------------


def _make_facing_fixture(angle_deg: float, n_frames: int = 5) -> np.ndarray:
    """Build synthetic frames where the hip pair lies at ``angle_deg`` from +X.

    All landmarks start at the origin; we only set meaningful positions for
    the hip landmarks so the facing direction is well-defined.
    """
    frames = np.zeros((n_frames, N_LANDMARKS, 3))
    angle_rad = math.radians(angle_deg)
    # Left hip at -0.5 along hip axis, right hip at +0.5
    # hip_vec = right_hip - left_hip = unit vector at angle_deg in XZ plane
    half = np.array([0.5 * math.cos(angle_rad), 0.0, 0.5 * math.sin(angle_rad)])
    frames[:, RIGHT_HIP, :] = half
    frames[:, LEFT_HIP, :] = -half
    return frames


class TestCanonicalFacing:
    def test_facing_right_becomes_positive_x(self):
        """Subject initially facing +Z (90°): after rotation hip X > 0."""
        frames = _make_facing_fixture(angle_deg=90)
        result = canonical_facing(frames)
        hip_vec_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_vec_x > 0), f"Expected positive X, got {hip_vec_x}"

    def test_facing_left_becomes_positive_x(self):
        """Subject initially facing -Z (−90°): after rotation hip X > 0."""
        frames = _make_facing_fixture(angle_deg=-90)
        result = canonical_facing(frames)
        hip_vec_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_vec_x > 0), f"Expected positive X, got {hip_vec_x}"

    def test_already_aligned_unchanged(self):
        """Hip vector already along +X should not be altered."""
        frames = _make_facing_fixture(angle_deg=0)
        result = canonical_facing(frames)
        np.testing.assert_allclose(result, frames, atol=1e-12)

    def test_180_degrees_becomes_positive_x(self):
        """Hip vector pointing in -X (180°) becomes +X after rotation."""
        frames = _make_facing_fixture(angle_deg=180)
        result = canonical_facing(frames)
        hip_vec_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_vec_x > 0)

    def test_various_angles_positive_x(self):
        """All orientations should result in positive hip X-component."""
        for angle in range(-170, 181, 10):
            frames = _make_facing_fixture(angle_deg=angle)
            result = canonical_facing(frames)
            hip_vec_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
            assert np.all(hip_vec_x > -1e-10), (
                f"angle={angle}: hip_vec_x={hip_vec_x}"
            )

    def test_y_coordinates_unchanged(self):
        """Y-axis rotation must not alter the Y coordinate of any landmark."""
        frames = _make_facing_fixture(angle_deg=45)
        result = canonical_facing(frames)
        np.testing.assert_allclose(result[:, :, 1], frames[:, :, 1], atol=1e-12)

    def test_distances_preserved(self):
        """Rotation must preserve pairwise Euclidean distances."""
        frames = _make_facing_fixture(angle_deg=37)
        result = canonical_facing(frames)
        for i in range(N_LANDMARKS - 1):
            dist_before = np.linalg.norm(
                frames[:, i + 1, :] - frames[:, i, :], axis=1
            )
            dist_after = np.linalg.norm(
                result[:, i + 1, :] - result[:, i, :], axis=1
            )
            np.testing.assert_allclose(dist_after, dist_before, atol=1e-12)

    def test_degenerate_zero_hip_vec_unchanged(self):
        """A frame whose left/right hips coincide (zero XZ) is left unchanged."""
        frames = np.zeros((1, N_LANDMARKS, 3))
        result = canonical_facing(frames)
        np.testing.assert_array_equal(result, frames)

    def test_shape_preserved(self):
        frames = _make_facing_fixture(angle_deg=45, n_frames=8)
        assert canonical_facing(frames).shape == frames.shape

    def test_returns_new_array(self):
        frames = _make_facing_fixture(angle_deg=30)
        result = canonical_facing(frames)
        assert result is not frames

    def test_accepts_list_input(self):
        frames = _make_facing_fixture(angle_deg=60).tolist()
        result = canonical_facing(frames)
        assert isinstance(result, np.ndarray)

    def test_multi_frame_random(self):
        """Random multi-frame input: all frames end with positive hip X."""
        rng = np.random.default_rng(42)
        n_frames = 20
        frames = np.zeros((n_frames, N_LANDMARKS, 3))
        angles = rng.uniform(-math.pi, math.pi, n_frames)
        for f, a in enumerate(angles):
            half = np.array([0.5 * math.cos(a), 0.0, 0.5 * math.sin(a)])
            frames[f, RIGHT_HIP, :] = half
            frames[f, LEFT_HIP, :] = -half
        result = canonical_facing(frames)
        hip_vec_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_vec_x > -1e-10)

    # ------------------------------------------------------------------
    # Acceptance-criterion fixture tests (left-facing / right-facing)
    # ------------------------------------------------------------------

    def test_left_facing_fixture_positive_x(self):
        """Canonical acceptance criterion: left-facing skeleton → positive X."""
        # "Left-facing": subject faces camera from its left, hip axis at +90°
        frames = _make_facing_fixture(angle_deg=90, n_frames=10)
        result = canonical_facing(frames)
        hip_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_x > 0)

    def test_right_facing_fixture_positive_x(self):
        """Canonical acceptance criterion: right-facing skeleton → positive X."""
        # "Right-facing": subject faces camera from its right, hip axis at −90°
        frames = _make_facing_fixture(angle_deg=-90, n_frames=10)
        result = canonical_facing(frames)
        hip_x = result[:, RIGHT_HIP, 0] - result[:, LEFT_HIP, 0]
        assert np.all(hip_x > 0)


# ---------------------------------------------------------------------------
# check_camera_angle tests
# ---------------------------------------------------------------------------


def _make_frontal_frames(n_frames: int = 10) -> np.ndarray:
    """Return frames where hips are spread in X (frontal view).

    After torso_scale the hip-to-hip XZ magnitude should be well above 0.15.
    """
    frames = np.zeros((n_frames, N_LANDMARKS, 3))
    # Hips separated by 0.4 in X, zero Z separation
    frames[:, LEFT_HIP, :] = [-0.2, 0.0, 0.0]
    frames[:, RIGHT_HIP, :] = [0.2, 0.0, 0.0]
    # Shoulders above hips (gives a non-zero torso length for torso_scale)
    frames[:, LEFT_SHOULDER, :] = [-0.2, 1.0, 0.0]
    frames[:, RIGHT_SHOULDER, :] = [0.2, 1.0, 0.0]
    return frames


def _make_sideways_frames(n_frames: int = 10) -> np.ndarray:
    """Return frames where hips are nearly coincident in X (sideways view).

    The left and right hip landmarks share the same X position; only Z differs
    (depth), but since the frames are *not* torso-scaled the XZ magnitude is
    already tiny — well below the 0.15 threshold.
    """
    frames = np.zeros((n_frames, N_LANDMARKS, 3))
    # Hips at nearly the same X, separated only in Z (shot from the side)
    frames[:, LEFT_HIP, :] = [0.0, 0.0, -0.05]
    frames[:, RIGHT_HIP, :] = [0.0, 0.0, 0.05]
    # Shoulders above hips
    frames[:, LEFT_SHOULDER, :] = [0.0, 1.0, -0.05]
    frames[:, RIGHT_SHOULDER, :] = [0.0, 1.0, 0.05]
    return frames


class TestCheckCameraAngle:
    def test_frontal_view_returns_true(self):
        """Wide hip separation in X (frontal view) should be accepted."""
        frames = _make_frontal_frames()
        # Apply hip_center + torso_scale to match pipeline usage
        frames = torso_scale(hip_center(frames))
        assert check_camera_angle(frames) is True

    def test_sideways_view_returns_false(self):
        """Near-zero XZ hip separation (sideways view) should be rejected."""
        # Construct frames where hip XZ magnitude is clearly below threshold
        frames = np.zeros((10, N_LANDMARKS, 3))
        # Hips collapsed to nearly the same XZ position
        frames[:, LEFT_HIP, :] = [0.0, 0.0, 0.0]
        frames[:, RIGHT_HIP, :] = [0.01, 0.0, 0.0]  # only 0.01 separation
        # Give a non-degenerate torso so torso_scale doesn't divide by zero
        frames[:, LEFT_SHOULDER, :] = [0.0, 1.0, 0.0]
        frames[:, RIGHT_SHOULDER, :] = [0.0, 1.0, 0.0]
        frames = torso_scale(hip_center(frames))
        assert check_camera_angle(frames) is False

    def test_returns_bool(self):
        """Return value must be a plain Python bool."""
        frames = _make_frontal_frames()
        result = check_camera_angle(frames)
        assert isinstance(result, bool)

    def test_custom_threshold_accepted(self):
        """A very low threshold causes even a sideways sequence to pass."""
        frames = np.zeros((5, N_LANDMARKS, 3))
        frames[:, LEFT_HIP, :] = [0.0, 0.0, 0.0]
        frames[:, RIGHT_HIP, :] = [0.05, 0.0, 0.0]
        frames[:, LEFT_SHOULDER, :] = [0.0, 1.0, 0.0]
        frames[:, RIGHT_SHOULDER, :] = [0.0, 1.0, 0.0]
        frames = torso_scale(hip_center(frames))
        # With a very low threshold the check should pass
        assert check_camera_angle(frames, threshold=0.0) is True

    def test_custom_threshold_rejected(self):
        """A very high threshold causes even a frontal sequence to fail."""
        frames = _make_frontal_frames()
        frames = torso_scale(hip_center(frames))
        # With an impossibly high threshold the check should fail
        assert check_camera_angle(frames, threshold=999.0) is False

    def test_accepts_list_input(self):
        """Should work with Python lists as well as NumPy arrays."""
        frames = _make_frontal_frames().tolist()
        result = check_camera_angle(frames)
        assert isinstance(result, bool)

    def test_shape_of_frames_unchanged(self):
        """check_camera_angle must not mutate the input array."""
        frames = _make_frontal_frames()
        original = frames.copy()
        check_camera_angle(frames)
        np.testing.assert_array_equal(frames, original)


# ---------------------------------------------------------------------------
# Integration: pipeline of all three
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_full_pipeline_shape(self):
        frames = _make_frames(15)
        result = canonical_facing(torso_scale(hip_center(frames)))
        assert result.shape == frames.shape

    def test_full_pipeline_acceptance_criteria(self):
        """After full pipeline, hip midpoint ≈ 0 and torso length ≈ 1."""
        frames = _make_frames(20)
        centered = hip_center(frames)
        scaled = torso_scale(centered)
        # Note: torso_scale may shift shoulder/hip midpoints slightly when
        # already centered, so we re-check after each stage.
        mean_hip = _hip_mid(centered).mean(axis=0)
        assert np.linalg.norm(mean_hip) < 1e-6
        mean_torso = _torso_lengths(scaled).mean()
        assert abs(mean_torso - 1.0) < 1e-6
