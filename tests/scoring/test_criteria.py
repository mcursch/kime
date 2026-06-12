"""
Unit tests for backend.scoring.criteria.

The primary acceptance criterion is: a landmark sequence that is identical
to the reference template must score >= 0.95 for every criterion (because all
raw deltas are zero, mapping to score = 1.0).

Additional tests verify the CriterionResult contract (field types, score
bounds, non-empty unit) and basic sensitivity (a deliberately degraded input
scores lower than the reference-identical baseline).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.scoring.criteria import (
    CHAMBER_HEIGHT_TOLERANCE,
    EXTENSION_ANGLE_TOLERANCE,
    CriterionResult,
    balance,
    chamber_height,
    extension_angle,
    guard_position,
    hip_rotation,
    retraction_speed,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (duplicated locally to keep tests self-contained)
# ---------------------------------------------------------------------------
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

SCORERS = [
    chamber_height,
    hip_rotation,
    extension_angle,
    balance,
    guard_position,
    retraction_speed,
]


# ---------------------------------------------------------------------------
# Fixture: a realistic reference sequence
# ---------------------------------------------------------------------------


def _make_front_kick_sequence(T: int = 30) -> np.ndarray:
    """
    Build a deterministic landmark sequence (shape T×33×3) that simulates a
    front-kick motion:  chamber → extension → retraction.

    Coordinate convention: hip-centred, y-up, z-forward, scale = torso-length.
    """
    seq = np.zeros((T, 33, 3), dtype=np.float64)

    # ---- static landmarks ------------------------------------------------
    # Shoulders
    seq[:, _LEFT_SHOULDER,  :] = [-0.30,  0.80, 0.00]
    seq[:, _RIGHT_SHOULDER, :] = [ 0.30,  0.80, 0.00]
    # Hips (origin ≈ [±0.15, 0])
    seq[:, _LEFT_HIP,  :] = [-0.15, 0.00, 0.00]
    seq[:, _RIGHT_HIP, :] = [ 0.15, 0.00, 0.00]
    # Nose
    seq[:, _NOSE, :] = [0.00, 1.20, 0.10]
    # Support foot (left, stationary)
    seq[:, _LEFT_ANKLE,  :] = [-0.15, -1.00, 0.00]
    seq[:, _LEFT_KNEE,   :] = [-0.15, -0.45, 0.00]
    # Arms: left wrist acts as guard (near nose), right arm extended
    seq[:, _LEFT_ELBOW,  :] = [-0.35, 0.50, 0.00]
    seq[:, _LEFT_WRIST,  :] = [-0.20, 0.90, 0.15]   # guard hand
    seq[:, _RIGHT_ELBOW, :] = [ 0.35, 0.50, 0.00]
    seq[:, _RIGHT_WRIST, :] = [ 0.40, 0.50, 0.20]

    # ---- right-leg kick motion -------------------------------------------
    phase_len = T // 3  # ~10 frames each

    for t in range(T):
        if t < phase_len:
            # Chamber: right knee rises
            progress = t / max(phase_len - 1, 1)
            knee_y = -0.30 + progress * 0.85   # -0.30 → +0.55
            seq[t, _RIGHT_KNEE,  :] = [ 0.15, knee_y,        0.00]
            seq[t, _RIGHT_ANKLE, :] = [ 0.15, knee_y - 0.45, 0.00]

        elif t < 2 * phase_len:
            # Extension: ankle thrusts forward
            progress = (t - phase_len) / max(phase_len - 1, 1)
            knee_y = 0.55 - progress * 0.10      # slight drop as leg straightens
            ankle_z = progress * 0.90            # forward thrust
            ankle_y = knee_y - 0.30 * (1.0 - 0.7 * progress)
            seq[t, _RIGHT_KNEE,  :] = [0.15, knee_y, 0.00]
            seq[t, _RIGHT_ANKLE, :] = [0.15, ankle_y, ankle_z]

        else:
            # Retraction: quick snap back to standing
            progress = (t - 2 * phase_len) / max(phase_len - 1, 1)
            knee_y   =  0.45 * (1.0 - progress) + (-0.45) * progress
            ankle_y  = (0.45 - 0.30) * (1.0 - progress) + (-1.00) * progress
            seq[t, _RIGHT_KNEE,  :] = [0.15, knee_y,  0.00]
            seq[t, _RIGHT_ANKLE, :] = [0.15, ankle_y, 0.90 * (1.0 - progress)]

    return seq


@pytest.fixture
def reference_sequence() -> np.ndarray:
    return _make_front_kick_sequence(T=30)


# ---------------------------------------------------------------------------
# Tests: reference-identical input → score >= 95
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_reference_identical_yields_high_score(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Passing the reference as both arguments must yield score >= 0.95."""
    ref = reference_sequence
    result = scorer(ref.copy(), ref)
    assert result.score >= 0.95, (
        f"{scorer.__name__}: expected score >= 0.95, got {result.score:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests: CriterionResult contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_returns_criterion_result(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Each scorer must return a CriterionResult instance."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert isinstance(result, CriterionResult)


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_score_in_bounds(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Score must always be in [0.0, 1.0]."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_unit_is_non_empty(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Unit string must be non-empty."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert isinstance(result.unit, str) and len(result.unit) > 0


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_name_matches_function(
    scorer, reference_sequence: np.ndarray
) -> None:
    """CriterionResult.name must equal the scorer function's __name__."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert result.name == scorer.__name__


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_delta_is_float(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Delta must be a Python float (not a numpy scalar)."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert isinstance(result.delta, float)


# ---------------------------------------------------------------------------
# Tests: reference-identical delta is (near) zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scorer", SCORERS, ids=[f.__name__ for f in SCORERS])
def test_reference_identical_delta_near_zero(
    scorer, reference_sequence: np.ndarray
) -> None:
    """Delta must be essentially zero when user == reference."""
    result = scorer(reference_sequence.copy(), reference_sequence)
    assert abs(result.delta) < 1e-9, (
        f"{scorer.__name__}: delta={result.delta} is not ~0 for identical inputs"
    )


# ---------------------------------------------------------------------------
# Tests: degraded input scores lower than reference-identical
# ---------------------------------------------------------------------------


def test_chamber_height_scores_lower_when_knee_too_low(
    reference_sequence: np.ndarray,
) -> None:
    """Dropping the knee significantly below the reference lowers the score."""
    ref = reference_sequence
    degraded = ref.copy()
    # Force both knees lower by more than the tolerance
    degraded[:, _LEFT_KNEE,  1] -= CHAMBER_HEIGHT_TOLERANCE + 0.10
    degraded[:, _RIGHT_KNEE, 1] -= CHAMBER_HEIGHT_TOLERANCE + 0.10

    perfect = chamber_height(ref.copy(), ref)
    bad     = chamber_height(degraded, ref)
    assert bad.score < perfect.score, (
        f"Expected degraded score {bad.score:.2f} < perfect score {perfect.score:.2f}"
    )


def test_extension_angle_scores_lower_when_limb_bent(
    reference_sequence: np.ndarray,
) -> None:
    """A severely bent striking limb at impact yields a lower extension score."""
    ref = reference_sequence
    degraded = ref.copy()
    # Collapse the right knee sharply inward to shrink the extension angle
    degraded[:, _RIGHT_KNEE, 1] -= EXTENSION_ANGLE_TOLERANCE * 0.5
    degraded[:, _RIGHT_KNEE, 2] += 0.50

    perfect = extension_angle(ref.copy(), ref)
    bad     = extension_angle(degraded, ref)
    assert bad.score < perfect.score, (
        f"Expected degraded score {bad.score:.2f} < perfect score {perfect.score:.2f}"
    )


def test_score_clamped_to_zero_on_extreme_degradation(
    reference_sequence: np.ndarray,
) -> None:
    """A catastrophically bad input must not produce a negative score."""
    ref = reference_sequence
    degraded = ref.copy()
    # Move the whole sequence wildly off reference
    degraded += 5.0
    for scorer in SCORERS:
        result = scorer(degraded, ref)
        assert result.score >= 0.0, (
            f"{scorer.__name__}: score {result.score:.4f} is negative"
        )


def test_score_capped_at_one(reference_sequence: np.ndarray) -> None:
    """Score must not exceed 1.0 even for an outperforming user."""
    ref = reference_sequence
    for scorer in SCORERS:
        result = scorer(ref.copy(), ref)
        assert result.score <= 1.0, (
            f"{scorer.__name__}: score {result.score:.4f} exceeds 1.0"
        )
