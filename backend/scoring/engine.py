"""
Public scoring engine for the Kime judging pipeline.

Entry point:  ``score_rep(technique, landmark_sequence) -> RepScore``

Pipeline (matches README §"How judging works"):
  1. DTW-align the user's landmark sequence to the expert reference template.
  2. Run each biomechanical criterion scorer on the aligned sequence.
  3. Assemble per-criterion ``CriterionResult`` objects.
  4. Compute the overall score as the weighted average defined in weights.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .weights import CRITERION_NAMES, TECHNIQUE_WEIGHTS
from .dtw_aligner import align_to_reference
from .criteria import score_all_criteria

# ── public data model ─────────────────────────────────────────────────────────

SUPPORTED_TECHNIQUES: frozenset[str] = frozenset(TECHNIQUE_WEIGHTS.keys())


@dataclass(frozen=True)
class CriterionResult:
    """Score and raw delta for a single biomechanical criterion.

    Attributes
    ----------
    name:
        Criterion identifier, e.g. ``"hip_rotation"``.
    score:
        Normalised score in **[0.0, 1.0]** — higher is better.
    delta:
        Signed numeric difference from the reference ideal value.
        Negative means below ideal; positive means above.
    weight:
        The fractional contribution of this criterion to the overall score
        for the technique being judged.
    """

    name: str
    score: float
    delta: float
    weight: float


@dataclass
class RepScore:
    """Aggregated score for a single rep.

    Attributes
    ----------
    technique:
        The martial arts technique that was judged.
    criteria:
        Exactly six ``CriterionResult`` objects, one per biomechanical criterion,
        in the canonical order defined by ``CRITERION_NAMES``.
    overall:
        Weighted average of per-criterion scores using weights from
        ``weights.TECHNIQUE_WEIGHTS[technique]``.
    """

    technique: str
    criteria: list[CriterionResult] = field(default_factory=list)
    overall: float = 0.0

    def __post_init__(self) -> None:
        if len(self.criteria) != len(CRITERION_NAMES):
            raise ValueError(
                f"RepScore requires exactly {len(CRITERION_NAMES)} criteria, "
                f"got {len(self.criteria)}"
            )


# ── public API ────────────────────────────────────────────────────────────────

def score_rep(technique: str, landmark_sequence: np.ndarray) -> RepScore:
    """Score a single rep of *technique* from its landmark sequence.

    Parameters
    ----------
    technique:
        One of ``"front_kick"``, ``"roundhouse_kick"``, ``"straight_punch"``.
    landmark_sequence:
        Float NumPy array of shape ``(T, 33, 3)`` — ``T`` frames × 33 MediaPipe
        landmarks × ``(x, y, z)`` normalised or metric coordinates.

    Returns
    -------
    RepScore
        Contains exactly six ``CriterionResult`` entries and an overall weighted
        average score.

    Raises
    ------
    ValueError
        If *technique* is not supported or *landmark_sequence* has an
        unexpected shape.
    """
    if technique not in SUPPORTED_TECHNIQUES:
        raise ValueError(
            f"Unknown technique {technique!r}. "
            f"Supported: {sorted(SUPPORTED_TECHNIQUES)}"
        )

    landmark_sequence = np.asarray(landmark_sequence, dtype=float)
    if landmark_sequence.ndim != 3 or landmark_sequence.shape[1:] != (33, 3):
        raise ValueError(
            f"landmark_sequence must have shape (T, 33, 3), "
            f"got {landmark_sequence.shape}"
        )

    # Step 1 — DTW alignment to expert reference template.
    aligned_seq = align_to_reference(technique, landmark_sequence)

    # Step 2 — Run each criterion scorer on the aligned sequence.
    raw_results: dict[str, tuple[float, float]] = score_all_criteria(
        technique, aligned_seq
    )

    # Step 3 — Assemble CriterionResult objects in canonical order.
    weights = TECHNIQUE_WEIGHTS[technique]
    criteria: list[CriterionResult] = [
        CriterionResult(
            name=name,
            score=float(raw_results[name][0]),
            delta=float(raw_results[name][1]),
            weight=weights[name],
        )
        for name in CRITERION_NAMES
    ]

    # Step 4 — Weighted average overall score.
    overall = float(
        sum(cr.score * cr.weight for cr in criteria)
    )

    return RepScore(technique=technique, criteria=criteria, overall=overall)
