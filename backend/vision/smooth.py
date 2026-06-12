"""Temporal smoothing of MediaPipe landmark sequences.

This module provides Savitzky-Golay filtering across the time axis of a
landmark array of shape (N_frames, 33, 3).

Default parameter rationale
---------------------------
window_length=9
    At 30 fps a 9-frame window covers ~300 ms, which is long enough to
    suppress high-frequency jitter from detector noise while being short
    enough to preserve the fast acceleration phases of martial-arts strikes
    (typical strike duration ≥ 150 ms).

polyorder=3
    A cubic polynomial fits the smooth, monotonic arcs that dominate limb
    trajectories.  Lower orders over-smooth sharp peaks; order 4+ starts to
    chase noise again without meaningful accuracy gain at window_length=9.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# Defaults exposed at module level so callers can reference them explicitly.
DEFAULT_WINDOW_LENGTH: int = 9
DEFAULT_POLYORDER: int = 3


def smooth_landmarks(
    landmarks: np.ndarray,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    polyorder: int = DEFAULT_POLYORDER,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter to a landmark sequence along the time axis.

    Parameters
    ----------
    landmarks:
        Array of shape ``(N_frames, 33, 3)`` containing the (x, y, z)
        coordinates of the 33 MediaPipe pose landmarks for each frame.
    window_length:
        Length of the filter window (number of frames).  Must be a positive
        odd integer greater than ``polyorder``.  Defaults to
        ``DEFAULT_WINDOW_LENGTH`` (9).
    polyorder:
        Order of the polynomial used to fit the samples.  Must be less than
        ``window_length``.  Defaults to ``DEFAULT_POLYORDER`` (3).

    Returns
    -------
    np.ndarray
        Smoothed array with the same shape as *landmarks*.

    Raises
    ------
    ValueError
        If ``landmarks.shape[0] < window_length`` — the sequence is too short
        for the requested window.
    ValueError
        If the array does not have exactly three dimensions, or if the last
        two dimensions are not ``(33, 3)``.
    """
    landmarks = np.asarray(landmarks, dtype=float)

    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 3):
        raise ValueError(
            f"landmarks must have shape (N_frames, 33, 3), got {landmarks.shape}"
        )

    n_frames = landmarks.shape[0]
    if n_frames < window_length:
        raise ValueError(
            f"Frame count ({n_frames}) is too short for window_length={window_length}. "
            "Provide more frames or reduce window_length."
        )

    # savgol_filter operates along axis=0 (time), smoothing each of the
    # 33*3 = 99 coordinate channels independently.
    smoothed: np.ndarray = savgol_filter(
        landmarks, window_length=window_length, polyorder=polyorder, axis=0
    )
    return smoothed
