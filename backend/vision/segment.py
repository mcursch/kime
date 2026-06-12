"""
Rep-window segmentation.

Locates the three key frames that define a single technique execution:

  chamber    — the moment the striking limb reaches its cocked/chambered position
  extension  — the moment of maximum extension (impact point)
  retraction — the moment the limb has returned toward guard

The approach uses the speed (frame-to-frame distance) of the primary joint(s)
for each movement type to find velocity peaks, then assigns the three landmarks
from those peaks in temporal order.

MediaPipe Pose landmark indices:
  15 – left wrist      16 – right wrist
  25 – left knee       26 – right knee
  27 – left ankle      28 – right ankle
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

# Primary joints to track per movement type.
# If a movement_type is not listed the union of all joints is used.
_JOINTS_BY_MOVEMENT: dict[str, list[int]] = {
    "front_kick":      [27, 28],  # ankles
    "roundhouse_kick": [27, 28],  # ankles
    "straight_punch":  [15, 16],  # wrists
}
_FALLBACK_JOINTS: list[int] = [15, 16, 27, 28]

# Minimum number of frames between detected peaks (avoids double-counting)
_MIN_PEAK_DISTANCE = 3


def find_rep_window(
    landmarks: np.ndarray,
    movement_type: str,
) -> tuple[int, int, int]:
    """Find (chamber, extension, retraction) frame indices for one rep.

    The function tracks the average speed of the primary joints for the given
    movement type, detects peaks in that speed signal, and maps them onto the
    three phases of the technique.

    Args:
        landmarks: (N, 33, 3) float array, normalised and smoothed.
        movement_type: One of "front_kick", "roundhouse_kick", "straight_punch"
            (or any string — unknown types fall back to all primary joints).

    Returns:
        A 3-tuple ``(chamber, extension, retraction)`` of integer frame indices
        satisfying ``0 <= chamber < extension < retraction < N``.

    Raises:
        ValueError: if *landmarks* has fewer than 4 frames (cannot compute
            meaningful peaks).
    """
    n_frames = landmarks.shape[0]
    if n_frames < 4:
        raise ValueError(
            f"find_rep_window needs at least 4 frames, got {n_frames}."
        )

    joint_indices = _JOINTS_BY_MOVEMENT.get(movement_type, _FALLBACK_JOINTS)

    # Mean position of selected joints across each frame → (N, 3)
    positions = landmarks[:, joint_indices, :].mean(axis=1)

    # Frame-to-frame speed → (N-1,)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    # Detect peaks; clamp minimum distance to avoid index-out-of-bounds
    min_dist = min(_MIN_PEAK_DISTANCE, max(1, n_frames // 5))
    peaks, properties = find_peaks(speed, distance=min_dist)

    # ---- Assign the three phases ----------------------------------------
    if len(peaks) == 0:
        # Flat or monotone signal: fall back to equal thirds
        chamber    = n_frames // 4
        extension  = n_frames // 2
        retraction = 3 * n_frames // 4
    elif len(peaks) == 1:
        # Single peak: treat as extension, bookend with equal spacing
        extension  = int(peaks[0])
        chamber    = max(0, extension // 2)
        retraction = min(n_frames - 1, extension + (n_frames - extension) // 2)
    else:
        # Multiple peaks: highest-speed peak → extension
        #   earliest peak before extension → chamber
        #   latest peak after extension → retraction (or next one after it)
        ext_idx   = int(np.argmax(speed[peaks]))
        extension = int(peaks[ext_idx])

        before = peaks[peaks < extension]
        after  = peaks[peaks > extension]

        chamber    = int(before[0])  if len(before) > 0 else max(0, extension - 1)
        retraction = int(after[-1])  if len(after)  > 0 else min(n_frames - 1, extension + 1)

    # Guarantee strict ordering (defensive clamp)
    chamber    = max(0,           min(chamber,    n_frames - 3))
    extension  = max(chamber + 1, min(extension,  n_frames - 2))
    retraction = max(extension + 1, min(retraction, n_frames - 1))

    return chamber, extension, retraction
