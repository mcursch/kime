"""
Rep-window segmentation via joint-velocity peak detection.

Given a normalised, smoothed pose sequence (N_frames × 33 × 3) and a
movement-type label this module locates the execution window of a single
striking repetition:

    chamber_frame  – loaded / pre-strike position (low velocity, before peak)
    extension_frame – strike at peak velocity
    retraction_frame – return / guard position (low velocity, after peak)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices used per movement type.
# See: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_KNEE = 25
_RIGHT_KNEE = 26

# Default set covers both hand and foot strikes.
_DEFAULT_LANDMARKS = [_LEFT_WRIST, _RIGHT_WRIST, _LEFT_ANKLE, _RIGHT_ANKLE]

# Per-technique subsets improve signal-to-noise for known movement types.
_LANDMARK_SETS: dict[str, list[int]] = {
    "straight_punch": [_LEFT_WRIST, _RIGHT_WRIST],
    "front_kick": [_LEFT_ANKLE, _RIGHT_ANKLE, _LEFT_KNEE, _RIGHT_KNEE],
    "roundhouse_kick": [_LEFT_ANKLE, _RIGHT_ANKLE, _LEFT_KNEE, _RIGHT_KNEE],
}

# Minimum number of frames required for a valid analysis.
_MIN_FRAMES = 5


class SegmentationError(RuntimeError):
    """Raised when a rep window cannot be reliably located in the pose sequence."""


def _joint_velocity(poses: np.ndarray, landmark_indices: list[int]) -> np.ndarray:
    """Return per-frame scalar velocity for the given landmark subset.

    Velocity at frame *t* is the mean Euclidean norm of the finite differences
    across all selected landmarks between frames *t-1* and *t*.  Frame 0 is
    assigned the same velocity as frame 1 so the output length matches
    *N_frames*.

    Parameters
    ----------
    poses:
        Float array of shape ``(N_frames, 33, 3)``.
    landmark_indices:
        Landmark indices to include in the velocity computation.

    Returns
    -------
    velocity:
        Float array of shape ``(N_frames,)``.
    """
    selected = poses[:, landmark_indices, :]  # (N, K, 3)
    diff = np.diff(selected, axis=0)          # (N-1, K, 3)
    norms = np.linalg.norm(diff, axis=-1)     # (N-1, K)
    frame_vel = norms.mean(axis=-1)           # (N-1,)
    # Prepend the first computed value so length == N_frames.
    return np.concatenate([[frame_vel[0]], frame_vel])


def find_rep_window(
    poses: np.ndarray,
    movement_type: str = "default",
) -> tuple[int, int, int]:
    """Locate the execution window of a single striking repetition.

    Parameters
    ----------
    poses:
        Normalised, smoothed pose landmarks of shape ``(N_frames, 33, 3)``.
        Coordinates should already be hip-centred and torso-normalised.
    movement_type:
        Technique label used to select the relevant landmark subset.
        Recognised values: ``"straight_punch"``, ``"front_kick"``,
        ``"roundhouse_kick"``.  Any other value falls back to the default
        wrist + ankle set.

    Returns
    -------
    chamber_frame:
        Frame index of the lowest-velocity frame **before** the velocity peak
        (loaded / pre-strike position).
    extension_frame:
        Frame index of the dominant velocity peak (strike at full extension).
    retraction_frame:
        Frame index of the lowest-velocity frame **after** the velocity peak
        (return / guard position).

    Raises
    ------
    SegmentationError
        If the sequence is too short, the velocity array is degenerate, or no
        clear peak can be detected.
    """
    poses = np.asarray(poses, dtype=float)

    if poses.ndim != 3 or poses.shape[1] != 33 or poses.shape[2] != 3:
        raise SegmentationError(
            f"poses must have shape (N_frames, 33, 3), got {poses.shape!r}"
        )

    n_frames = poses.shape[0]
    if n_frames < _MIN_FRAMES:
        raise SegmentationError(
            f"Sequence too short for segmentation: {n_frames} frames "
            f"(minimum {_MIN_FRAMES})"
        )

    landmark_indices = _LANDMARK_SETS.get(movement_type, _DEFAULT_LANDMARKS)
    velocity = _joint_velocity(poses, landmark_indices)

    vel_range = velocity.max() - velocity.min()
    if vel_range == 0.0:
        raise SegmentationError(
            "No detectable motion in pose sequence: velocity is flat "
            "(all frames have identical landmark positions)."
        )

    # Require peak height at least 20 % of the velocity range above the
    # minimum so that noise bumps on a near-flat trace are not selected.
    height_threshold = velocity.min() + 0.20 * vel_range

    peaks, properties = find_peaks(velocity, height=height_threshold)

    if peaks.size == 0:
        raise SegmentationError(
            "No velocity peak found in pose sequence.  The motion may be too "
            "uniform or the sequence may not contain a full striking rep "
            f"(movement_type={movement_type!r})."
        )

    # Select the dominant (highest) peak.
    extension_frame = int(peaks[np.argmax(properties["peak_heights"])])

    # Chamber: lowest-velocity frame strictly before the extension peak.
    # scipy.signal.find_peaks never returns index 0, so this slice is non-empty.
    chamber_frame = int(np.argmin(velocity[:extension_frame]))

    # Retraction: lowest-velocity frame strictly after the extension peak.
    # scipy.signal.find_peaks never returns the last index, so the tail is non-empty.
    tail_offset = extension_frame + 1
    retraction_frame = int(np.argmin(velocity[tail_offset:])) + tail_offset

    return chamber_frame, extension_frame, retraction_frame
