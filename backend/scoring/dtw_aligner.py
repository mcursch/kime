"""DTW alignment module for martial-arts technique scoring.

Wraps ``dtaidistance`` to align a normalized, smoothed landmark sequence
(shape ``(n_frames, n_features)``) against a per-technique reference
template loaded from ``backend/data/references/``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import List, Tuple

import numpy as np
from dtaidistance import dtw, dtw_ndim

# Directory that holds per-technique reference templates (.npy files).
_REFERENCES_DIR = pathlib.Path(__file__).parent.parent / "data" / "references"


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
    return np.load(template_path)


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
