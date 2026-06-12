"""
Rule-based biomechanical criterion scorers.

Each public function accepts an aligned landmark sequence
``(T, 33, 3)`` and returns ``(score, delta)`` where:

  score  — float in [0.0, 1.0], higher is better
  delta  — signed raw numeric difference from the ideal reference value
           (negative = below ideal, positive = above)

MediaPipe landmark indices (subset used here)
─────────────────────────────────────────────
 0  nose
11  left shoulder    12  right shoulder
13  left elbow       14  right elbow
15  left wrist       16  right wrist
23  left hip         24  right hip
25  left knee        26  right knee
27  left ankle       28  right ankle
29  left heel        30  right heel
31  left foot index  32  right foot index
"""

from __future__ import annotations

import numpy as np

# ── landmark index constants ──────────────────────────────────────────────────
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28
_L_HEEL, _R_HEEL = 29, 30
_L_FOOT, _R_FOOT = 31, 32

# ── helpers ───────────────────────────────────────────────────────────────────

def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at *b* formed by the vectors b→a and b→c (degrees)."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _sigmoid_score(value: float, ideal: float, scale: float) -> tuple[float, float]:
    """Map *abs(value - ideal)* to a [0, 1] score via a soft penalty curve.

    Returns ``(score, delta)`` where delta = value - ideal.
    """
    delta = value - ideal
    score = float(np.exp(-0.5 * (delta / (scale + 1e-9)) ** 2))
    return score, delta


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── per-criterion scorers ─────────────────────────────────────────────────────

def score_chamber_height(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score how high the striking limb is chambered.

    For kicks: max knee height relative to the ipsilateral hip.
    For punch: max wrist height relative to the ipsilateral shoulder.

    Ideal heights (normalised, positive = above reference joint):
      front_kick      — knee 0.15 above hip
      roundhouse_kick — knee 0.10 above hip
      straight_punch  — wrist at shoulder level (delta = 0)
    """
    ideals = {
        "front_kick": (0.15, 0.08),      # (ideal_delta, scale)
        "roundhouse_kick": (0.10, 0.08),
        "straight_punch": (0.00, 0.06),
    }
    ideal, scale = ideals.get(technique, (0.10, 0.08))

    if technique in ("front_kick", "roundhouse_kick"):
        # Use the higher knee (striking leg assumed to be the one moving more)
        l_knee_y = seq[:, _L_KNEE, 1]
        r_knee_y = seq[:, _R_KNEE, 1]
        l_hip_y = seq[:, _L_HIP, 1]
        r_hip_y = seq[:, _R_HIP, 1]
        # In image/normalised coords y increases downward; invert for "height"
        l_chamber = np.max(l_hip_y - l_knee_y)
        r_chamber = np.max(r_hip_y - r_knee_y)
        measured = float(max(l_chamber, r_chamber))
    else:
        # straight_punch — wrist vs shoulder
        l_delta = seq[:, _L_SHOULDER, 1] - seq[:, _L_WRIST, 1]
        r_delta = seq[:, _R_SHOULDER, 1] - seq[:, _R_WRIST, 1]
        measured = float(max(np.max(l_delta), np.max(r_delta)))

    return _sigmoid_score(measured, ideal, scale)


def score_hip_rotation(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score hip rotation at impact.

    Measured as the maximum signed rotation of the hip line away from the
    frontal plane during the sequence (degrees projected onto the XZ plane).

    Ideal rotations:
      front_kick      — 10 °  (hips stay mostly square)
      roundhouse_kick — 45 °  (large rotation for power)
      straight_punch  — 30 °  (moderate rotation)
    """
    ideals = {
        "front_kick": (10.0, 8.0),
        "roundhouse_kick": (45.0, 12.0),
        "straight_punch": (30.0, 10.0),
    }
    ideal, scale = ideals.get(technique, (20.0, 10.0))

    # Hip vector in XZ plane (z is depth)
    hip_vec = seq[:, _R_HIP, :] - seq[:, _L_HIP, :]  # (T, 3)
    # Angle relative to X axis in XZ plane
    angles = np.degrees(np.arctan2(hip_vec[:, 2], hip_vec[:, 0] + 1e-9))
    # Use range of rotation as the "rotation amount"
    rotation = float(np.max(np.abs(angles - angles[0])))

    return _sigmoid_score(rotation, ideal, scale)


def score_extension_angle(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score the joint angle at full extension.

    Kicks  — knee angle at the frame of maximum extension (ideal ≈ 170 °).
    Punch  — elbow angle at the frame of maximum extension (ideal ≈ 170 °).
    """
    ideals = {
        "front_kick": (170.0, 10.0),
        "roundhouse_kick": (165.0, 12.0),
        "straight_punch": (170.0, 8.0),
    }
    ideal, scale = ideals.get(technique, (170.0, 10.0))

    angles = []
    for frame in seq:
        if technique in ("front_kick", "roundhouse_kick"):
            ang_l = _angle_3pts(frame[_L_HIP], frame[_L_KNEE], frame[_L_ANKLE])
            ang_r = _angle_3pts(frame[_R_HIP], frame[_R_KNEE], frame[_R_ANKLE])
            angles.append(max(ang_l, ang_r))
        else:
            ang_l = _angle_3pts(frame[_L_SHOULDER], frame[_L_ELBOW], frame[_L_WRIST])
            ang_r = _angle_3pts(frame[_R_SHOULDER], frame[_R_ELBOW], frame[_R_WRIST])
            angles.append(max(ang_l, ang_r))

    measured = float(max(angles)) if angles else ideal
    return _sigmoid_score(measured, ideal, scale)


def score_balance(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score balance: how well the CoM stays over the support base.

    CoM approximated as the midpoint of hips.
    Support base approximated as the stance foot (the one that moves less).
    Score degrades as CoM lateral offset from foot exceeds the ideal (≈ 0 m).

    Ideal lateral offsets (normalised body units):
      all techniques — 0.0  (CoM directly over support foot)
    Scale reflects how much deviation is acceptable.
    """
    scales = {
        "front_kick": 0.15,
        "roundhouse_kick": 0.18,
        "straight_punch": 0.10,
    }
    scale = scales.get(technique, 0.12)
    ideal = 0.0

    com_x = (seq[:, _L_HIP, 0] + seq[:, _R_HIP, 0]) / 2.0

    # Support foot = the ankle that moves least (min std-dev in x)
    l_ankle_x = seq[:, _L_ANKLE, 0]
    r_ankle_x = seq[:, _R_ANKLE, 0]
    if float(np.std(l_ankle_x)) <= float(np.std(r_ankle_x)):
        support_x = l_ankle_x
    else:
        support_x = r_ankle_x

    offsets = np.abs(com_x - support_x)
    measured = float(np.mean(offsets))  # mean lateral drift

    score, delta = _sigmoid_score(measured, ideal, scale)
    return score, delta


def score_guard_position(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score the non-striking hand guard position.

    The guard hand should stay near the cheek/chin (elbow tucked).
    Measured as the mean distance of the non-striking wrist from the ipsilateral
    shoulder in the YZ plane (height + depth).

    Ideal distance (normalised):  0.20  (wrist close to shoulder/face)
    """
    ideals = {
        "front_kick": (0.20, 0.10),
        "roundhouse_kick": (0.20, 0.10),
        "straight_punch": (0.18, 0.08),
    }
    ideal, scale = ideals.get(technique, (0.20, 0.10))

    # For simplicity use the wrist that is NOT the dominant striking wrist.
    # We approximate by taking whichever wrist stays closer to its shoulder on
    # average (i.e. the guard hand rather than the striking hand).
    l_dist = np.linalg.norm(seq[:, _L_WRIST, :] - seq[:, _L_SHOULDER, :], axis=1)
    r_dist = np.linalg.norm(seq[:, _R_WRIST, :] - seq[:, _R_SHOULDER, :], axis=1)

    guard_dist = float(
        np.mean(l_dist) if np.mean(l_dist) < np.mean(r_dist) else np.mean(r_dist)
    )

    return _sigmoid_score(guard_dist, ideal, scale)


def score_retraction_speed(seq: np.ndarray, technique: str) -> tuple[float, float]:
    """Score retraction speed after impact.

    Fast retraction indicates control and readiness to follow up.
    Measured as the mean frame-to-frame speed of the striking limb landmark
    in the second half of the sequence (retraction phase).

    Ideal speeds (normalised units / frame):
      front_kick      — 0.04
      roundhouse_kick — 0.05
      straight_punch  — 0.06
    """
    ideals = {
        "front_kick": (0.04, 0.02),
        "roundhouse_kick": (0.05, 0.02),
        "straight_punch": (0.06, 0.025),
    }
    ideal, scale = ideals.get(technique, (0.04, 0.02))

    T = len(seq)
    if T < 4:
        return 0.5, 0.0

    half = T // 2
    retraction = seq[half:]

    if technique in ("front_kick", "roundhouse_kick"):
        # Use the ankle that moves most (striking foot)
        l_speed = np.mean(np.linalg.norm(np.diff(retraction[:, _L_ANKLE, :], axis=0), axis=1))
        r_speed = np.mean(np.linalg.norm(np.diff(retraction[:, _R_ANKLE, :], axis=0), axis=1))
        measured = float(max(l_speed, r_speed))
    else:
        l_speed = np.mean(np.linalg.norm(np.diff(retraction[:, _L_WRIST, :], axis=0), axis=1))
        r_speed = np.mean(np.linalg.norm(np.diff(retraction[:, _R_WRIST, :], axis=0), axis=1))
        measured = float(max(l_speed, r_speed))

    return _sigmoid_score(measured, ideal, scale)


# ── public dispatcher ─────────────────────────────────────────────────────────

# Maps criterion name -> scorer function
_SCORERS: dict[str, object] = {
    "chamber_height": score_chamber_height,
    "hip_rotation": score_hip_rotation,
    "extension_angle": score_extension_angle,
    "balance": score_balance,
    "guard_position": score_guard_position,
    "retraction_speed": score_retraction_speed,
}


def score_all_criteria(
    technique: str,
    aligned_seq: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Run every criterion scorer and return ``{name: (score, delta)}``."""
    from .weights import CRITERION_NAMES  # avoid circular at module level

    results: dict[str, tuple[float, float]] = {}
    for name in CRITERION_NAMES:
        scorer = _SCORERS[name]
        results[name] = scorer(aligned_seq, technique)  # type: ignore[operator]
    return results
