"""Tests for backend.vision.smooth.smooth_landmarks."""

import numpy as np
import pytest

from backend.vision.smooth import (
    DEFAULT_POLYORDER,
    DEFAULT_WINDOW_LENGTH,
    smooth_landmarks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LANDMARKS = 33
N_COORDS = 3
RNG = np.random.default_rng(42)


def _make_landmarks(n_frames: int, *, noise_std: float = 0.0) -> np.ndarray:
    """Return a (n_frames, 33, 3) array filled with zeros (+ optional noise)."""
    base = np.zeros((n_frames, N_LANDMARKS, N_COORDS))
    if noise_std > 0.0:
        base += RNG.normal(scale=noise_std, size=base.shape)
    return base


def _sinusoid_landmarks(
    n_frames: int, freq: float = 1.0, fps: float = 30.0, noise_std: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, ground_truth) landmark arrays shaped (n_frames, 33, 3).

    A sinusoid at *freq* Hz (sampled at *fps*) is broadcast to every
    landmark coordinate.  White noise with std *noise_std* is added to
    produce the noisy version.
    """
    t = np.linspace(0, n_frames / fps, n_frames)
    signal = np.sin(2 * np.pi * freq * t)  # (n_frames,)
    # Broadcast to (n_frames, 33, 3)
    ground_truth = np.broadcast_to(signal[:, None, None], (n_frames, N_LANDMARKS, N_COORDS)).copy()
    noise = RNG.normal(scale=noise_std, size=ground_truth.shape)
    noisy = ground_truth + noise
    return noisy, ground_truth


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_shape_matches_input_default_params(self):
        data = _make_landmarks(60)
        result = smooth_landmarks(data)
        assert result.shape == data.shape

    def test_shape_matches_input_custom_params(self):
        data = _make_landmarks(50)
        result = smooth_landmarks(data, window_length=7, polyorder=2)
        assert result.shape == data.shape

    def test_output_dtype_is_float(self):
        data = _make_landmarks(30).astype(np.float32)
        result = smooth_landmarks(data)
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# ValueError: sequence too short
# ---------------------------------------------------------------------------


class TestTooShortRaisesValueError:
    def test_exact_window_minus_one_raises(self):
        short = _make_landmarks(DEFAULT_WINDOW_LENGTH - 1)
        with pytest.raises(ValueError, match="too short"):
            smooth_landmarks(short)

    def test_one_frame_raises(self):
        with pytest.raises(ValueError, match="too short"):
            smooth_landmarks(_make_landmarks(1))

    def test_custom_window_too_short_raises(self):
        data = _make_landmarks(4)
        with pytest.raises(ValueError, match="too short"):
            smooth_landmarks(data, window_length=5, polyorder=2)

    def test_exactly_window_length_does_not_raise(self):
        data = _make_landmarks(DEFAULT_WINDOW_LENGTH)
        result = smooth_landmarks(data)  # Should not raise
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# ValueError: bad array shape
# ---------------------------------------------------------------------------


class TestBadShapeRaisesValueError:
    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="shape"):
            smooth_landmarks(np.zeros((60, 99)))

    def test_wrong_landmarks_count_raises(self):
        with pytest.raises(ValueError, match="shape"):
            smooth_landmarks(np.zeros((60, 17, 3)))

    def test_wrong_coords_count_raises(self):
        with pytest.raises(ValueError, match="shape"):
            smooth_landmarks(np.zeros((60, 33, 2)))

    def test_4d_array_raises(self):
        with pytest.raises(ValueError, match="shape"):
            smooth_landmarks(np.zeros((60, 33, 3, 1)))


# ---------------------------------------------------------------------------
# Smoothing quality: ≥30% RMS reduction on sinusoid + noise
# ---------------------------------------------------------------------------


class TestSmoothingQuality:
    def test_rms_reduction_at_least_30_percent(self):
        n_frames = 120  # 4 s at 30 fps — enough cycles for the filter to act
        noisy, ground_truth = _sinusoid_landmarks(
            n_frames, freq=2.0, fps=30.0, noise_std=0.1
        )
        smoothed = smooth_landmarks(noisy)

        rms_before = float(np.sqrt(np.mean((noisy - ground_truth) ** 2)))
        rms_after = float(np.sqrt(np.mean((smoothed - ground_truth) ** 2)))

        reduction = (rms_before - rms_after) / rms_before
        assert reduction >= 0.30, (
            f"Expected ≥30% RMS reduction, got {reduction:.1%} "
            f"(before={rms_before:.4f}, after={rms_after:.4f})"
        )

    def test_smooth_constant_signal_unchanged(self):
        """A perfectly constant signal should be returned unchanged."""
        data = np.ones((60, N_LANDMARKS, N_COORDS)) * 3.14
        result = smooth_landmarks(data)
        np.testing.assert_allclose(result, data, atol=1e-10)


# ---------------------------------------------------------------------------
# Default constants are sensible
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_window_length_value(self):
        assert DEFAULT_WINDOW_LENGTH == 9

    def test_default_polyorder_value(self):
        assert DEFAULT_POLYORDER == 3

    def test_polyorder_less_than_window(self):
        assert DEFAULT_POLYORDER < DEFAULT_WINDOW_LENGTH
