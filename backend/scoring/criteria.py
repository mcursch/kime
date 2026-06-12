"""
Rule-based biomechanical criteria scorers for martial-arts technique analysis.

Landmark array convention
--------------------------
Shape : (T, 33, 3)  — frames × landmarks × (x, y, z)
Origin: mid-hip (hip-centred after normalisation)
Scale : torso length  (shoulder-midpoint to hip-midpoint = 1.0)
Axes  : x right, y up, z forward  (world-coordinate convention after
        canonical rotation)

Each public scorer accepts:

    aligned_seq : np.ndarray, shape (T, 33, 3)
        Warping-path-aligned user landmark sequence.
    reference   : np.ndarray, shape (T_ref, 33, 3)
        Expert reference template (same normalisation convention).

and returns a :class:`CriterionResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (33-point model)
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

# ---------------------------------------------------------------------------
# Tolerance constants  (|delta| at which the score reaches 0)
# These define the "full-miss" threshold for each criterion.
# ---------------------------------------------------------------------------

#: Maximum acceptable deficit in chamber height before score reaches 0.
#: Units: normalised torso-lengths.
CHAMBER_HEIGHT_TOLERANCE: float = 0.40

#: Maximum acceptable deviation in hip yaw at impact before score reaches 0.
#: Units: degrees.
HIP_ROTATION_TOLERANCE: float = 45.0

#: Maximum acceptable deviation in striking-limb extension angle.
#: Units: degrees.
EXTENSION_ANGLE_TOLERANCE: float = 45.0

#: Maximum acceptable lateral CoM offset from the support foot.
#: Units: normalised torso-lengths.
BALANCE_COM_TOLERANCE: float = 0.30

#: Maximum acceptable displacement of the guard wrist from the reference.
#: Units: normalised torso-lengths.
GUARD_POSITION_TOLERANCE: float = 0.40

#: Maximum acceptable deficit in retraction speed before score reaches 0.
#: Units: normalised torso-lengths per frame.
RETRACTION_SPEED_TOLERANCE: float = 0.20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    """Score for a single biomechanical criterion."""

    name: str
    """Criterion identifier (snake_case)."""

    score: float
    """Technique quality for this criterion, in the range [0.0, 1.0]."""

    delta: float
    """User metric minus reference metric, expressed in *unit*."""

    unit: str
    """Human-readable unit for *delta* (non-empty)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _angle_at_joint(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
) -> float:
    """Return the interior angle (degrees) at *joint* in the proximal–joint–distal chain."""
    v1 = proximal - joint
    v2 = distal - joint
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _hip_yaw_deg(seq: np.ndarray) -> np.ndarray:
    """
    Yaw rotation of the hip-to-hip vector in the x–z plane, per frame.

    Returns an array of shape (T,) in degrees.  A frontal-facing pose has
    yaw ≈ 0°; rotating the hips into a strike produces a non-zero yaw.
    """
    vec = seq[:, _RIGHT_HIP, :] - seq[:, _LEFT_HIP, :]   # (T, 3)
    return np.degrees(np.arctan2(vec[:, 2], vec[:, 0]))   # (T,)


def _active_end_effector_index(seq: np.ndarray) -> int:
    """Return the landmark index of the most-displaced end-effector.

    Checks both wrists and both ankles; returns the one with the highest
    peak displacement from its position in frame 0.  This heuristic
    identifies the striking limb without needing an explicit technique label.
    """
    candidates = [_LEFT_WRIST, _RIGHT_WRIST, _LEFT_ANKLE, _RIGHT_ANKLE]
    peak_displacements = [
        float(np.max(np.linalg.norm(seq[:, idx, :] - seq[0, idx, :], axis=-1)))
        for idx in candidates
    ]
    return candidates[int(np.argmax(peak_displacements))]


def _impact_frame(seq: np.ndarray, ee_idx: int) -> int:
    """Return the index of the impact frame (peak end-effector displacement)."""
    disp = np.linalg.norm(seq[:, ee_idx, :] - seq[0, ee_idx, :], axis=-1)
    return int(np.argmax(disp))


def _score_from_delta(delta: float, tolerance: float) -> float:
    """Map a raw delta to a [0.0, 1.0] score via linear normalisation."""
    return float(max(0.0, 1.0 - abs(delta) / tolerance))


# ---------------------------------------------------------------------------
# Public scorer functions
# ---------------------------------------------------------------------------


def chamber_height(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Peak height of the raised knee during the chamber phase.

    Measures the maximum y-coordinate (up direction) reached by either knee
    across the full motion sequence.  Higher peak knee height indicates
    better chamber mechanics and greater potential striking power.

    Delta is positive when the user chambers higher than the reference and
    negative when the user's chamber is too low.
    """
    user_peak = float(np.max(aligned_seq[:, [_LEFT_KNEE, _RIGHT_KNEE], 1]))
    ref_peak  = float(np.max(reference[:,   [_LEFT_KNEE, _RIGHT_KNEE], 1]))

    delta = user_peak - ref_peak
    score = _score_from_delta(delta, CHAMBER_HEIGHT_TOLERANCE)
    return CriterionResult(
        name="chamber_height",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="torso-lengths",
    )


def hip_rotation(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Hip yaw (rotation about the vertical axis) at the moment of impact.

    The impact frame is identified as the frame of peak active end-effector
    displacement.  Greater hip rotation transfers more kinetic energy into
    the strike; a square-hip posture at impact indicates under-rotation.

    Delta is the difference (user yaw − reference yaw) in degrees.
    """
    ee_idx = _active_end_effector_index(aligned_seq)

    impact_user = _impact_frame(aligned_seq, ee_idx)
    impact_ref  = _impact_frame(reference,   ee_idx)

    user_yaw = float(_hip_yaw_deg(aligned_seq)[impact_user])
    ref_yaw  = float(_hip_yaw_deg(reference) [impact_ref])

    delta = user_yaw - ref_yaw
    score = _score_from_delta(delta, HIP_ROTATION_TOLERANCE)
    return CriterionResult(
        name="hip_rotation",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="degrees",
    )


def extension_angle(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Interior angle of the striking limb at the moment of impact.

    For kicks the knee angle (hip–knee–ankle) is measured; for punches the
    elbow angle (shoulder–elbow–wrist) is measured.  The striking limb is
    chosen automatically via :func:`_active_end_effector_index`.  Full
    extension approaches 180°; a bent limb at impact indicates incomplete
    technique.

    Delta is user angle − reference angle (degrees).
    """
    ee_idx = _active_end_effector_index(aligned_seq)

    # Select the joint chain that corresponds to the active end-effector.
    if ee_idx == _LEFT_WRIST:
        proximal_idx, joint_idx, distal_idx = _LEFT_SHOULDER,  _LEFT_ELBOW,  _LEFT_WRIST
    elif ee_idx == _RIGHT_WRIST:
        proximal_idx, joint_idx, distal_idx = _RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST
    elif ee_idx == _LEFT_ANKLE:
        proximal_idx, joint_idx, distal_idx = _LEFT_HIP,  _LEFT_KNEE,  _LEFT_ANKLE
    else:  # _RIGHT_ANKLE
        proximal_idx, joint_idx, distal_idx = _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE

    impact_user = _impact_frame(aligned_seq, ee_idx)
    impact_ref  = _impact_frame(reference,   ee_idx)

    user_angle = _angle_at_joint(
        aligned_seq[impact_user, proximal_idx, :],
        aligned_seq[impact_user, joint_idx,    :],
        aligned_seq[impact_user, distal_idx,   :],
    )
    ref_angle = _angle_at_joint(
        reference[impact_ref, proximal_idx, :],
        reference[impact_ref, joint_idx,    :],
        reference[impact_ref, distal_idx,   :],
    )

    delta = user_angle - ref_angle
    score = _score_from_delta(delta, EXTENSION_ANGLE_TOLERANCE)
    return CriterionResult(
        name="extension_angle",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="degrees",
    )


def balance(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Lateral offset of the centre of mass over the support foot.

    The centre of mass is estimated as the mean position of the four hip
    and shoulder landmarks.  The support foot is identified as the ankle
    with the lower mean y-coordinate (i.e. nearer the ground).  Smaller
    lateral offset indicates better balance throughout the technique.

    Delta is user offset − reference offset (torso-lengths); positive means
    the user leans further from the support foot than the reference.
    """
    core = [_LEFT_HIP, _RIGHT_HIP, _LEFT_SHOULDER, _RIGHT_SHOULDER]

    def _lateral_offset(seq: np.ndarray) -> float:
        com_x = float(np.mean(seq[:, core, 0]))
        left_y  = float(np.mean(seq[:, _LEFT_ANKLE,  1]))
        right_y = float(np.mean(seq[:, _RIGHT_ANKLE, 1]))
        support_x = (
            float(np.mean(seq[:, _LEFT_ANKLE,  0]))
            if left_y <= right_y
            else float(np.mean(seq[:, _RIGHT_ANKLE, 0]))
        )
        return float(abs(com_x - support_x))

    user_offset = _lateral_offset(aligned_seq)
    ref_offset  = _lateral_offset(reference)

    delta = user_offset - ref_offset
    score = _score_from_delta(delta, BALANCE_COM_TOLERANCE)
    return CriterionResult(
        name="balance",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="torso-lengths",
    )


def guard_position(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Distance of the guarding wrist from the nose (guard-hand proximity).

    The guard hand is the wrist that maintains the shorter average distance
    to the nose across the sequence.  A guard held close to the face
    (within reference range) signals proper defensive positioning; a dropped
    guard registers as a positive delta.

    Delta is user guard distance − reference guard distance (torso-lengths).
    """
    def _guard_dist(seq: np.ndarray) -> float:
        nose  = seq[:, _NOSE, :]
        left  = float(np.mean(np.linalg.norm(seq[:, _LEFT_WRIST,  :] - nose, axis=-1)))
        right = float(np.mean(np.linalg.norm(seq[:, _RIGHT_WRIST, :] - nose, axis=-1)))
        return min(left, right)

    user_dist = _guard_dist(aligned_seq)
    ref_dist  = _guard_dist(reference)

    delta = user_dist - ref_dist
    score = _score_from_delta(delta, GUARD_POSITION_TOLERANCE)
    return CriterionResult(
        name="guard_position",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="torso-lengths",
    )


def retraction_speed(
    aligned_seq: np.ndarray,
    reference: np.ndarray,
) -> CriterionResult:
    """
    Mean frame-to-frame speed of the active end-effector during retraction.

    The retraction phase is defined as all frames following the impact frame
    (peak displacement of the active end-effector).  Fast retraction
    indicates crisp technique and reduces counter-attack exposure.

    Delta is user retraction speed − reference retraction speed
    (torso-lengths per frame); negative delta means the user retracts too
    slowly.
    """
    ee_idx = _active_end_effector_index(aligned_seq)

    def _mean_retract_speed(seq: np.ndarray) -> float:
        ee = seq[:, ee_idx, :]
        impact = _impact_frame(seq, ee_idx)
        post_impact = ee[impact:]
        if len(post_impact) < 2:
            return 0.0
        return float(np.mean(np.linalg.norm(np.diff(post_impact, axis=0), axis=-1)))

    user_speed = _mean_retract_speed(aligned_seq)
    ref_speed  = _mean_retract_speed(reference)

    delta = user_speed - ref_speed
    score = _score_from_delta(delta, RETRACTION_SPEED_TOLERANCE)
    return CriterionResult(
        name="retraction_speed",
        score=round(score, 4),
        delta=round(delta, 6),
        unit="torso-lengths/frame",
    )


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------


def score_all_criteria(
    technique: str,
    aligned_seq: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Run all six biomechanical criterion scorers and return results as a dict.

    The reference template for *technique* is loaded from disk so that per-
    criterion deltas are computed relative to the expert reference.

    Parameters
    ----------
    technique:
        Technique slug, e.g. ``"front_kick"``.
    aligned_seq:
        User landmark sequence that has already been DTW-aligned to the
        reference.  Shape ``(T, 33, 3)``.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping of canonical criterion name → ``(score, delta)`` where
        *score* is in **[0.0, 1.0]** and *delta* retains the physical units
        defined by each criterion scorer.

    Raises
    ------
    FileNotFoundError
        If the reference template for *technique* does not exist.
    """
    # Import here to avoid a circular import between criteria ↔ dtw_aligner.
    from backend.scoring.dtw_aligner import load_reference_template  # noqa: PLC0415

    ref_flat = load_reference_template(technique)  # (n_frames, 99)
    reference = ref_flat.reshape(ref_flat.shape[0], 33, 3)

    raw_results: list[CriterionResult] = [
        chamber_height(aligned_seq, reference),
        hip_rotation(aligned_seq, reference),
        extension_angle(aligned_seq, reference),
        balance(aligned_seq, reference),
        guard_position(aligned_seq, reference),
        retraction_speed(aligned_seq, reference),
    ]

    return {cr.name: (cr.score, cr.delta) for cr in raw_results}
