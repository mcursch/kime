"""DTW alignment module for martial-arts technique scoring.

Wraps ``dtaidistance`` to align a normalized, smoothed landmark sequence
(shape ``(n_frames, n_features)``) against a per-technique reference
template loaded from ``backend/data/references/``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import List, Tuple

import numpy as np
from dtaidistance import dtw, dtw_ndim

# Directory that holds per-technique reference templates (.npy files).
_REFERENCES_DIR = pathlib.Path(__file__).parent.parent / "data" / "references"

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    """Result returned by :func:`align_sequence`.

    Attributes
    ----------
    path:
        Warping path as a list of ``(query_frame_idx, reference_frame_idx)``
        index pairs that define the optimal alignment.
    frame_distances:
        Per-step Euclidean distance between the aligned query frame and its
        matched reference frame.  Length equals ``len(path)``.
    mean_distance:
        Mean of *frame_distances* — a scalar summary of alignment quality.
        Lower values indicate a closer match to the reference.
    """

    path: List[Tuple[int, int]]
    frame_distances: np.ndarray
    mean_distance: float


def load_reference_template(technique_slug: str) -> np.ndarray:
    """Load the reference landmark template for *technique_slug*.

    Parameters
    ----------
    technique_slug:
        Identifier that matches a ``.npy`` file under
        ``backend/data/references/``, e.g. ``"front_kick"``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, n_features)`` containing the reference
        motion template.

    Raises
    ------
    FileNotFoundError
        If no ``.npy`` file named *technique_slug* exists in the references
        directory.
    """
    template_path = _REFERENCES_DIR / f"{technique_slug}.npy"
    if not template_path.exists():
        raise FileNotFoundError(
            f"No reference template found for technique '{technique_slug}'. "
            f"Expected file: {template_path}. "
            f"Available templates: {sorted(p.stem for p in _REFERENCES_DIR.glob('*.npy'))}."
        )

    # Emit a warning when the template is a known synthetic stub so that
    # operators can see at a glance that Phase 2 data has not been integrated.
    # The sidecar is written by scripts/generate_reference_templates.py and
    # should be removed (along with the .npy) once real pipeline output lands.
    meta_path = template_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        if meta.get("source") == "synthetic":
            logger.warning(
                "Loading SYNTHETIC reference template for '%s'. "
                "This is a schematic development stub — replace it with "
                "real Phase 2 pipeline output (see LIN-168).",
                technique_slug,
            )

    return np.load(template_path)


def align_to_reference(
    technique: str,
    landmark_sequence: np.ndarray,
) -> np.ndarray:
    """Align *landmark_sequence* to the expert reference template for *technique*.

    This is the high-level entry point used by the scoring engine.  It loads
    the reference template, runs DTW alignment, and returns the user's frames
    reordered according to the optimal warping path so that every returned
    frame is phase-matched to a reference frame.

    Parameters
    ----------
    technique:
        Technique slug, e.g. ``"front_kick"``.  Must have a corresponding
        ``.npy`` file in ``backend/data/references/``.
    landmark_sequence:
        Normalized, smoothed user pose sequence, shape ``(T, 33, 3)``.

    Returns
    -------
    np.ndarray
        Warped user sequence, shape ``(L, 33, 3)`` where *L* is the length of
        the DTW warping path.  Each frame ``aligned[k]`` is the user frame
        that best matches reference frame ``path[k][1]``.

    Raises
    ------
    FileNotFoundError
        If the reference template for *technique* does not exist.
    ValueError
        If *landmark_sequence* is not a valid 3-D array with 33 landmarks and
        3 coordinates.
    """
    landmark_sequence = np.asarray(landmark_sequence, dtype=np.double)
    if landmark_sequence.ndim != 3 or landmark_sequence.shape[1:] != (33, 3):
        raise ValueError(
            f"landmark_sequence must have shape (T, 33, 3), "
            f"got {landmark_sequence.shape}."
        )

    reference = load_reference_template(technique)  # (n_frames, 99)

    # Flatten (T, 33, 3) -> (T, 99) for the DTW distance computation.
    T = landmark_sequence.shape[0]
    flat_seq = landmark_sequence.reshape(T, -1)  # (T, 99)

    result = align_sequence(flat_seq, reference)

    # Reorder user frames according to the query indices in the warping path.
    query_indices = [i for i, _j in result.path]
    aligned = landmark_sequence[query_indices]  # (len(path), 33, 3)
    return aligned


def align_sequence(
    sequence: np.ndarray,
    reference: np.ndarray,
) -> AlignmentResult:
    """Align *sequence* to *reference* using multi-dimensional DTW.

    Both arrays must have the same number of feature columns (axis 1).  The
    number of frames (axis 0) may differ; DTW handles variable-length
    sequences automatically.

    Parameters
    ----------
    sequence:
        Query landmark sequence, shape ``(n_frames, n_features)``.
        Typically produced by the normalization + smoothing pipeline:
        33 landmarks × 3 coordinates = 99 features per frame.
    reference:
        Reference template loaded via :func:`load_reference_template`,
        shape ``(m_frames, n_features)``.

    Returns
    -------
    AlignmentResult
        Contains the warping path, per-step distances, and scalar mean
        distance.

    Raises
    ------
    ValueError
        If *sequence* or *reference* are not 2-D arrays, or if their feature
        dimensions do not match.
    """
    sequence = np.asarray(sequence, dtype=np.double)
    reference = np.asarray(reference, dtype=np.double)

    if sequence.ndim != 2:
        raise ValueError(
            f"sequence must be a 2-D array of shape (n_frames, n_features), "
            f"got shape {sequence.shape}."
        )
    if reference.ndim != 2:
        raise ValueError(
            f"reference must be a 2-D array of shape (n_frames, n_features), "
            f"got shape {reference.shape}."
        )
    if sequence.shape[1] != reference.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: sequence has {sequence.shape[1]} "
            f"features but reference has {reference.shape[1]}."
        )

    # Compute the accumulated-cost matrix and extract the optimal warping path.
    _dtw_dist, paths_matrix = dtw_ndim.warping_paths(sequence, reference)
    path: List[Tuple[int, int]] = dtw.best_path(paths_matrix)

    # Per-step Euclidean distance between matched frames.
    frame_distances = np.array(
        [np.linalg.norm(sequence[i] - reference[j]) for i, j in path],
        dtype=np.double,
    )

    mean_distance: float = float(frame_distances.mean())

    return AlignmentResult(
        path=path,
        frame_distances=frame_distances,
        mean_distance=mean_distance,
    )
