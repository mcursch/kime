"""
Preprocessing pipeline entry point.

``preprocess`` is the single public function in this module.  It accepts a raw
landmark array and a movement type string, runs the full normalisation →
smoothing → segmentation chain, and returns a ``PreprocessResult`` dataclass
that the DTW alignment step consumes downstream.

This file owns **no** business logic.  Every transformation is delegated to:
  - backend.vision.normalize  (hip_center, torso_scale, canonical_facing)
  - backend.vision.smooth     (smooth_landmarks)
  - backend.vision.segment    (find_rep_window)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.vision.normalize import (
    canonical_facing,
    check_camera_angle,
    hip_center,
    torso_scale,
)
from backend.vision.segment import find_rep_window
from backend.vision.smooth import smooth_landmarks

_MIN_FRAMES = 10


@dataclass
class PreprocessResult:
    """Holds the output of the full preprocessing pipeline.

    Attributes:
        landmarks: (N, 33, 3) float array — normalised and smoothed.
        chamber: Frame index of the limb-chambered phase.
        extension: Frame index of maximum extension (impact point).
        retraction: Frame index where the limb has returned toward guard.
        camera_angle_ok: False when the hip-vector XZ magnitude is too small,
            indicating the subject was filmed from the side rather than the
            front.  Scores produced from a sideways view are likely unreliable.
    """

    landmarks: np.ndarray
    chamber: int
    extension: int
    retraction: int
    camera_angle_ok: bool


def preprocess(
    landmarks: np.ndarray,
    movement_type: str,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on a raw landmark sequence.

    Steps applied in order:
      1. hip_center        — translate each frame so the hip midpoint is the origin
      2. torso_scale       — scale each frame so torso length == 1.0
      3. canonical_facing  — rotate the sequence to a canonical facing direction
      4. smooth_landmarks  — apply Savitzky-Golay temporal smoothing
      5. find_rep_window   — locate (chamber, extension, retraction) frame indices

    Args:
        landmarks: (N, 33, 3) float array of raw MediaPipe landmarks where N is
            the number of frames, 33 is the number of pose landmarks, and 3 are
            the (x, y, z) coordinates.
        movement_type: Technique identifier, e.g. ``"front_kick"``,
            ``"roundhouse_kick"``, or ``"straight_punch"``.  Passed unchanged
            to ``find_rep_window`` to select the primary joints for segmentation.

    Returns:
        A :class:`PreprocessResult` containing the normalised+smoothed landmark
        array and the three integer frame indices.

    Raises:
        ValueError: if *landmarks* has fewer than :data:`_MIN_FRAMES` frames.
            This check runs before any other step so callers receive a clear
            error rather than a cryptic failure inside a downstream module.
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 3):
        raise ValueError(
            f"landmarks must have shape (N, 33, 3); got {landmarks.shape}."
        )

    n_frames = landmarks.shape[0]
    if n_frames < _MIN_FRAMES:
        raise ValueError(
            f"landmarks must contain at least {_MIN_FRAMES} frames for a "
            f"meaningful analysis; got {n_frames}."
        )

    # --- Normalisation ---
    lm = hip_center(landmarks)
    lm = torso_scale(lm)

    # Camera-angle quality gate: check before canonical_facing so we operate on
    # the raw (unrotated) scaled data, though the result is identical either
    # way because canonical_facing preserves XZ magnitude.
    camera_angle_ok = check_camera_angle(lm)

    lm = canonical_facing(lm)

    # --- Smoothing ---
    lm = smooth_landmarks(lm)

    # --- Segmentation ---
    chamber, extension, retraction = find_rep_window(lm, movement_type)

    return PreprocessResult(
        landmarks=lm,
        chamber=chamber,
        extension=extension,
        retraction=retraction,
        camera_angle_ok=camera_angle_ok,
    )
