"""
DTW alignment module for the Kime scoring engine.

``align_to_reference`` warps the user's landmark sequence to the time axis of
an expert reference template using Dynamic Time Warping, so that slower or
faster execution does not unfairly penalise form.

Phase 4 of the development plan (see README) will populate the reference
template store.  Until then, the function returns the input sequence unchanged
(identity warp) so the rest of the pipeline remains exercisable.

Expected array shape for *landmark_sequence*:
    (num_frames, 33, 3)  — T frames × 33 MediaPipe landmarks × (x, y, z)

The function always returns an array of the same shape as the input; callers
must not assume the output length equals the reference length at this stage.
"""

from __future__ import annotations

import numpy as np

# Registry of reference templates: technique -> np.ndarray | None.
# Populated by the offline reference-building pipeline.
_REFERENCE_TEMPLATES: dict[str, np.ndarray | None] = {
    "front_kick": None,
    "roundhouse_kick": None,
    "straight_punch": None,
}


def align_to_reference(
    technique: str,
    landmark_sequence: np.ndarray,
) -> np.ndarray:
    """Return *landmark_sequence* warped onto the reference time axis.

    Parameters
    ----------
    technique:
        One of ``"front_kick"``, ``"roundhouse_kick"``, ``"straight_punch"``.
    landmark_sequence:
        Float array of shape ``(T, 33, 3)``.

    Returns
    -------
    np.ndarray
        Aligned array.  Shape is ``(T_ref, 33, 3)`` when a reference is
        available, otherwise ``(T, 33, 3)`` (identity pass-through).
    """
    if landmark_sequence.ndim != 3 or landmark_sequence.shape[1:] != (33, 3):
        raise ValueError(
            f"landmark_sequence must have shape (T, 33, 3), "
            f"got {landmark_sequence.shape}"
        )

    reference = _REFERENCE_TEMPLATES.get(technique)
    if reference is None:
        # No reference available yet — return the sequence unchanged.
        return landmark_sequence.copy()

    # Full DTW implementation (dtaidistance or equivalent) will go here once
    # reference templates exist.  The flattened per-frame feature vector is
    # used as the DTW signal: shape (T, 99).
    user_flat = landmark_sequence.reshape(len(landmark_sequence), -1)
    ref_flat = reference.reshape(len(reference), -1)

    try:
        from dtaidistance import dtw_ndim  # type: ignore[import]

        path = dtw_ndim.warping_path(user_flat, ref_flat)
        # Resample user frames along the warping path to match reference length.
        ref_indices = [j for _, j in path]
        # For each reference frame, pick the closest warped user frame.
        aligned_frames: list[np.ndarray] = []
        for ref_idx in range(len(reference)):
            # Find user indices that map to this reference index.
            user_indices = [i for i, j in path if j == ref_idx]
            frame = landmark_sequence[user_indices].mean(axis=0)
            aligned_frames.append(frame)
        return np.stack(aligned_frames)
    except ImportError:
        # dtaidistance not yet installed — graceful degradation.
        return landmark_sequence.copy()
