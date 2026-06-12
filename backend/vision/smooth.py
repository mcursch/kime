"""
Temporal smoothing for landmark sequences.

Uses a Savitzky-Golay filter (scipy.signal.savgol_filter) applied along the
time axis independently for each landmark and each spatial coordinate.  The
filter parameters are chosen conservatively so they work on short sequences
while still attenuating high-frequency MediaPipe jitter.
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_landmarks(
    landmarks: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth a landmark sequence along the time axis with a Savitzky-Golay filter.

    Args:
        landmarks: (N, 33, 3) float array.
        window_length: Nominal SG window length (odd integer).  Automatically
            reduced if the sequence is shorter than the requested window.
        polyorder: Polynomial order for the SG filter.  Must be less than
            *window_length*.

    Returns:
        (N, 33, 3) smoothed float array.

    Raises:
        ValueError: if *polyorder* >= *window_length* after adjustment.
    """
    n_frames = landmarks.shape[0]

    # Clamp window to the actual sequence length; keep it odd.
    wl = min(window_length, n_frames)
    if wl % 2 == 0:
        wl -= 1

    # Ensure window is large enough for the polynomial order.
    min_wl = polyorder + 1
    if min_wl % 2 == 0:
        min_wl += 1
    wl = max(wl, min_wl)

    if wl > n_frames:
        raise ValueError(
            f"Cannot smooth: adjusted window_length ({wl}) exceeds the number "
            f"of frames ({n_frames}).  Use a shorter sequence or smaller polyorder."
        )

    return savgol_filter(landmarks, window_length=wl, polyorder=polyorder, axis=0)
