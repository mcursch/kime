"""
Tests for backend.scoring.engine.score_rep.

Covers:
  - All three supported techniques produce a RepScore.
  - Each RepScore has exactly six CriterionResult entries.
  - Criterion names match the canonical CRITERION_NAMES list.
  - All per-criterion scores are in [0.0, 1.0].
  - The overall score equals the manually recomputed weighted average.
  - CriterionResult weights match TECHNIQUE_WEIGHTS.
  - Weights of all criteria in a RepScore sum to 1.0.
  - Unsupported technique raises ValueError.
  - Malformed landmark_sequence raises ValueError.
  - DTW aligner is called (integration: result is unchanged when aligner
    returns a pass-through).
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from backend.scoring.engine import (
    CriterionResult,
    RepScore,
    score_rep,
    SUPPORTED_TECHNIQUES,
)
from backend.scoring.weights import CRITERION_NAMES, TECHNIQUE_WEIGHTS

# ── fixtures ──────────────────────────────────────────────────────────────────

_TECHNIQUES = sorted(SUPPORTED_TECHNIQUES)


def _make_seq(num_frames: int = 30, seed: int = 0) -> np.ndarray:
    """Return a plausible (T, 33, 3) landmark array.

    Landmarks are placed in anatomically rough positions so that angle and
    distance calculations yield sensible (non-degenerate) values.
    """
    rng = np.random.default_rng(seed)

    # Canonical T-pose skeleton (x, y, z) in normalised body units.
    # y increases downward (image convention).
    base = np.zeros((33, 3), dtype=float)

    # Head
    base[0] = [0.00, -1.80, 0.0]   # nose

    # Shoulders
    base[11] = [-0.25, -1.50, 0.0]  # left shoulder
    base[12] = [0.25, -1.50, 0.0]   # right shoulder

    # Elbows
    base[13] = [-0.45, -1.20, 0.0]  # left elbow
    base[14] = [0.45, -1.20, 0.0]   # right elbow

    # Wrists
    base[15] = [-0.45, -0.90, 0.0]  # left wrist
    base[16] = [0.45, -0.90, 0.0]   # right wrist

    # Hips
    base[23] = [-0.12, 0.00, 0.0]   # left hip
    base[24] = [0.12, 0.00, 0.0]    # right hip

    # Knees
    base[25] = [-0.12, 0.50, 0.0]   # left knee
    base[26] = [0.12, 0.50, 0.0]    # right knee

    # Ankles
    base[27] = [-0.12, 1.00, 0.0]   # left ankle
    base[28] = [0.12, 1.00, 0.0]    # right ankle

    # Heels
    base[29] = [-0.13, 1.05, -0.05]
    base[30] = [0.13, 1.05, -0.05]

    # Foot indices
    base[31] = [-0.10, 1.05, 0.10]
    base[32] = [0.10, 1.05, 0.10]

    # Broadcast to (T, 33, 3) with slight per-frame noise
    seq = np.tile(base, (num_frames, 1, 1))
    seq += rng.normal(0, 0.01, seq.shape)
    return seq


# ── core acceptance-criteria tests ───────────────────────────────────────────

@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_score_rep_returns_rep_score(technique: str) -> None:
    seq = _make_seq()
    result = score_rep(technique, seq)
    assert isinstance(result, RepScore)


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_exactly_six_criteria(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    assert len(result.criteria) == 6, (
        f"{technique}: expected 6 criteria, got {len(result.criteria)}"
    )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_criterion_names_match_canonical(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    names = [cr.name for cr in result.criteria]
    assert names == list(CRITERION_NAMES), (
        f"{technique}: criterion names {names!r} != canonical {list(CRITERION_NAMES)!r}"
    )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_criterion_scores_in_range(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    for cr in result.criteria:
        assert 0.0 <= cr.score <= 1.0, (
            f"{technique}/{cr.name}: score {cr.score} out of [0, 1]"
        )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_overall_is_weighted_average(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    expected = sum(cr.score * cr.weight for cr in result.criteria)
    assert math.isclose(result.overall, expected, rel_tol=1e-9), (
        f"{technique}: overall {result.overall} != weighted avg {expected}"
    )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_criterion_weights_match_weights_py(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    defined = TECHNIQUE_WEIGHTS[technique]
    for cr in result.criteria:
        assert math.isclose(cr.weight, defined[cr.name], rel_tol=1e-9), (
            f"{technique}/{cr.name}: weight {cr.weight} != {defined[cr.name]}"
        )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_weights_sum_to_one(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    total = sum(cr.weight for cr in result.criteria)
    assert math.isclose(total, 1.0, rel_tol=1e-9), (
        f"{technique}: weights sum to {total}"
    )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_overall_score_in_range(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    assert 0.0 <= result.overall <= 1.0, (
        f"{technique}: overall score {result.overall} out of [0, 1]"
    )


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_technique_stored_on_rep_score(technique: str) -> None:
    result = score_rep(technique, _make_seq())
    assert result.technique == technique


# ── error-handling tests ──────────────────────────────────────────────────────

def test_unsupported_technique_raises() -> None:
    with pytest.raises(ValueError, match="Unknown technique"):
        score_rep("spinning_heel_kick", _make_seq())


def test_wrong_shape_raises_2d() -> None:
    with pytest.raises(ValueError, match="shape"):
        score_rep("front_kick", np.zeros((30, 99)))


def test_wrong_shape_raises_bad_landmarks() -> None:
    with pytest.raises(ValueError, match="shape"):
        score_rep("front_kick", np.zeros((30, 17, 3)))


def test_wrong_shape_raises_bad_coords() -> None:
    with pytest.raises(ValueError, match="shape"):
        score_rep("front_kick", np.zeros((30, 33, 2)))


# ── integration: DTW aligner is invoked ──────────────────────────────────────

def test_dtw_aligner_is_called() -> None:
    seq = _make_seq()
    with patch(
        "backend.scoring.engine.align_to_reference",
        wraps=lambda tech, s: s.copy(),
    ) as mock_align:
        score_rep("straight_punch", seq)
    mock_align.assert_called_once()
    call_tech, call_seq = mock_align.call_args[0]
    assert call_tech == "straight_punch"
    assert call_seq.shape == seq.shape


# ── per-technique smoke tests (different seeds) ───────────────────────────────

@pytest.mark.parametrize("technique", _TECHNIQUES)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_deterministic_across_seeds(technique: str, seed: int) -> None:
    """Same sequence produces the same RepScore each time."""
    seq = _make_seq(seed=seed)
    r1 = score_rep(technique, seq)
    r2 = score_rep(technique, seq)
    assert r1.overall == r2.overall
    for c1, c2 in zip(r1.criteria, r2.criteria):
        assert c1.score == c2.score
        assert c1.delta == c2.delta


# ── RepScore construction guard ───────────────────────────────────────────────

def test_rep_score_wrong_criteria_count_raises() -> None:
    with pytest.raises(ValueError, match="exactly"):
        RepScore(technique="front_kick", criteria=[], overall=0.0)
