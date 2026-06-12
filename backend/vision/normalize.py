"""
Landmark normalization utilities for MediaPipe 33-landmark pose arrays.

All functions accept and return NumPy arrays of shape (N_frames, 33, 3),
where the 3 channels are (x, y, z).

MediaPipe landmark indices used here:
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
"""

import numpy as np

# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Minimum mean XZ hip-vector magnitude (in torso-scale units) below which the
# subject is considered to be filmed from the side rather than the front/back.
# At torso-scale == 1, a frontal view yields a hip-width of ~0.4; sideways
# views produce values close to zero because the monocular depth (Z) estimate
# is unreliable and the left/right hip landmarks collapse to the same X column.
_CAMERA_ANGLE_THRESHOLD = 0.15


def check_camera_angle(
    frames: np.ndarray,
    threshold: float = _CAMERA_ANGLE_THRESHOLD,
) -> bool:
    """Return *True* when the camera angle is acceptable for scoring.

    Computes the XZ-plane magnitude of the hip vector (right_hip − left_hip)
    for every frame and checks whether the mean exceeds *threshold*.  A low
    mean magnitude indicates that both hip landmarks are collapsing to the same
    position in the XZ plane — the hallmark of a sideways filming angle where
    MediaPipe's monocular depth is unreliable.

    This check should be applied **after** :func:`torso_scale` so that the
    magnitude is expressed in normalised torso-length units and is therefore
    independent of subject size or camera distance.  Because :func:`canonical_facing`
    performs a rotation around the Y-axis it preserves XZ magnitude, so the
    check may equivalently be applied before or after that step.

    Parameters
    ----------
    frames : np.ndarray, shape (N_frames, 33, 3)
        Landmark coordinates (should be torso-scaled before calling).
    threshold : float
        Minimum acceptable mean XZ hip-vector magnitude.  Defaults to
        :data:`_CAMERA_ANGLE_THRESHOLD` (0.15 torso-lengths).

    Returns
    -------
    bool
        ``True``  — camera angle is acceptable (frontal or near-frontal).
        ``False`` — subject appears to be filmed sideways; scores may be
                    unreliable.
    """
    frames = np.asarray(frames, dtype=float)
    hip_vec = frames[:, RIGHT_HIP, :] - frames[:, LEFT_HIP, :]
    dx = hip_vec[:, 0]
    dz = hip_vec[:, 2]
    xz_mag = np.sqrt(dx ** 2 + dz ** 2)
    return bool(np.mean(xz_mag) >= threshold)


def hip_center(frames: np.ndarray) -> np.ndarray:
    """Translate each frame so the hip midpoint is at the origin.

    The hip midpoint is defined as the mean of the left-hip (index 23) and
    right-hip (index 24) landmarks.  The same translation is applied to all
    33 landmarks within each frame independently.

    Parameters
    ----------
    frames : np.ndarray, shape (N_frames, 33, 3)
        Raw landmark coordinates.

    Returns
    -------
    np.ndarray, shape (N_frames, 33, 3)
        Translated landmark coordinates.
    """
    frames = np.array(frames, dtype=float)
    # Hip midpoint per frame, shape (N_frames, 3)
    hip_mid = (frames[:, LEFT_HIP, :] + frames[:, RIGHT_HIP, :]) / 2.0
    # Broadcast-subtract: (N_frames, 1, 3)
    return frames - hip_mid[:, np.newaxis, :]


def torso_scale(frames: np.ndarray) -> np.ndarray:
    """Scale each frame so the torso length equals 1.

    Torso length is the Euclidean distance from the shoulder midpoint
    (mean of landmarks 11 and 12) to the hip midpoint (mean of landmarks
    23 and 24).  Each frame is scaled independently by its own torso length.
    Frames whose torso length is zero are left unchanged to avoid division
    by zero.

    Parameters
    ----------
    frames : np.ndarray, shape (N_frames, 33, 3)
        Landmark coordinates (typically already hip-centred).

    Returns
    -------
    np.ndarray, shape (N_frames, 33, 3)
        Scaled landmark coordinates.
    """
    frames = np.array(frames, dtype=float)
    shoulder_mid = (frames[:, LEFT_SHOULDER, :] + frames[:, RIGHT_SHOULDER, :]) / 2.0
    hip_mid = (frames[:, LEFT_HIP, :] + frames[:, RIGHT_HIP, :]) / 2.0

    # Torso length per frame, shape (N_frames,)
    torso_len = np.linalg.norm(shoulder_mid - hip_mid, axis=1)

    # Avoid division by zero; keep degenerate frames unchanged
    safe_len = np.where(torso_len == 0.0, 1.0, torso_len)

    # Divide all landmarks: (N_frames, 33, 3) / (N_frames, 1, 1)
    return frames / safe_len[:, np.newaxis, np.newaxis]


def canonical_facing(frames: np.ndarray) -> np.ndarray:
    """Rotate each frame around the Y-axis so the left-to-right hip vector
    points in the +X direction.

    "Facing" is derived from the hip orientation: the vector pointing from the
    left hip (index 23) to the right hip (index 24) is projected onto the XZ
    plane and rotated to align with the +X axis.  This makes the signed
    hip-to-hip X-component positive regardless of whether the subject was
    originally facing left or right.

    Each frame is rotated independently using its own hip orientation.
    Frames whose hip vector has zero XZ magnitude are left unchanged.

    Parameters
    ----------
    frames : np.ndarray, shape (N_frames, 33, 3)
        Landmark coordinates (typically hip-centred and torso-scaled).

    Returns
    -------
    np.ndarray, shape (N_frames, 33, 3)
        Rotationally normalised landmark coordinates.
    """
    frames = np.array(frames, dtype=float)

    # Vector from left hip to right hip, projected onto XZ plane
    hip_vec = frames[:, RIGHT_HIP, :] - frames[:, LEFT_HIP, :]  # (N_frames, 3)
    dx = hip_vec[:, 0]  # X component
    dz = hip_vec[:, 2]  # Z component

    # Angle of the hip vector in the XZ plane relative to +X axis
    theta = np.arctan2(dz, dx)  # (N_frames,)

    # Rotation by -theta around Y aligns the hip vector to +X:
    #   x' =  x * cos(theta) + z * sin(theta)
    #   y' =  y  (unchanged)
    #   z' = -x * sin(theta) + z * cos(theta)
    cos_t = np.cos(theta)  # (N_frames,)
    sin_t = np.sin(theta)  # (N_frames,)

    # Skip rotation for frames with zero hip XZ magnitude (degenerate frames)
    xz_mag = np.sqrt(dx ** 2 + dz ** 2)
    mask = (xz_mag != 0.0).astype(float)  # 1.0 = rotate, 0.0 = identity

    eff_cos = mask * cos_t + (1.0 - mask)  # identity cos = 1
    eff_sin = mask * sin_t                  # identity sin = 0

    x = frames[:, :, 0]  # (N_frames, 33)
    y = frames[:, :, 1]
    z = frames[:, :, 2]

    # Broadcast rotation coefficients over landmarks: (N_frames, 1)
    c = eff_cos[:, np.newaxis]
    s = eff_sin[:, np.newaxis]

    x_rot = x * c + z * s
    z_rot = -x * s + z * c

    return np.stack([x_rot, y, z_rot], axis=2)
