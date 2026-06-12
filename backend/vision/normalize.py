"""
Skeleton normalization transforms.

Each function takes an (N_frames, 33, 3) landmark array and returns a new
array of the same shape with the requested normalization applied.  No
in-place mutation — callers receive independent copies.

MediaPipe Pose landmark indices used here:
  11 – left shoulder
  12 – right shoulder
  23 – left hip
  24 – right hip
"""

import numpy as np

# MediaPipe Pose landmark indices
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24


def hip_center(landmarks: np.ndarray) -> np.ndarray:
    """Translate all landmarks so the hip midpoint sits at the origin each frame.

    Args:
        landmarks: (N, 33, 3) float array.

    Returns:
        (N, 33, 3) array translated per-frame so the midpoint of the left and
        right hip landmarks is (0, 0, 0).
    """
    out = landmarks.copy()
    hip_mid = (out[:, _LEFT_HIP, :] + out[:, _RIGHT_HIP, :]) / 2.0  # (N, 3)
    out -= hip_mid[:, np.newaxis, :]
    return out


def torso_scale(landmarks: np.ndarray) -> np.ndarray:
    """Scale each frame so the torso length (hip-centre → shoulder-centre) equals 1.

    Args:
        landmarks: (N, 33, 3) float array, ideally already hip-centred.

    Returns:
        (N, 33, 3) array uniformly scaled per-frame.

    Raises:
        ValueError: if any frame has a degenerate (near-zero) torso length.
    """
    out = landmarks.copy()
    hip_mid = (out[:, _LEFT_HIP, :] + out[:, _RIGHT_HIP, :]) / 2.0        # (N, 3)
    shoulder_mid = (out[:, _LEFT_SHOULDER, :] + out[:, _RIGHT_SHOULDER, :]) / 2.0  # (N, 3)
    torso_len = np.linalg.norm(shoulder_mid - hip_mid, axis=1)             # (N,)

    if np.any(torso_len < 1e-6):
        raise ValueError(
            "One or more frames have a near-zero torso length; "
            "cannot scale.  Check that hip and shoulder landmarks are present."
        )

    out /= torso_len[:, np.newaxis, np.newaxis]
    return out


def canonical_facing(landmarks: np.ndarray) -> np.ndarray:
    """Rotate landmarks in the XZ plane so the subject faces +Z on average.

    The rotation is computed from the *mean* hip axis across all frames so
    that a single rigid rotation is applied to the whole sequence (frame-wise
    rotation would destroy meaningful motion patterns).

    Args:
        landmarks: (N, 33, 3) float array, ideally already hip-centred and
            torso-scaled.

    Returns:
        (N, 33, 3) array rotated so the mean left→right hip vector points
        along +X, making the mean facing direction +Z.
    """
    out = landmarks.copy()

    # Left-to-right hip vector averaged over all frames
    hip_axis = out[:, _RIGHT_HIP, :] - out[:, _LEFT_HIP, :]   # (N, 3)
    mean_axis = hip_axis.mean(axis=0)                           # (3,)

    # Angle of the hip axis in the XZ plane
    angle = np.arctan2(mean_axis[2], mean_axis[0])

    # Rotation that brings mean_axis onto the +X axis (so facing is +Z)
    rot_angle = -angle
    cos_a = np.cos(rot_angle)
    sin_a = np.sin(rot_angle)

    # Rotation matrix around the Y axis (XZ plane rotation)
    rot = np.array([
        [ cos_a, 0.0, sin_a],
        [ 0.0,   1.0, 0.0  ],
        [-sin_a, 0.0, cos_a],
    ])  # (3, 3)

    # Apply: (N, 33, 3) @ (3, 3)^T  → broadcast over frames and landmarks
    out = out @ rot.T
    return out
