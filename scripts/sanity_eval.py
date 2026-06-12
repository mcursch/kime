#!/usr/bin/env python3
"""
scripts/sanity_eval.py — Kime judging-engine sanity evaluation.

Purpose
-------
Loads bundled synthetic landmark sequences and runs each through score_rep(),
then prints a structured pass/fail table:

    Expert clips  must score >= EXPERT_THRESHOLD (default 75).
    Bad clips     must score <= BAD_CEILING      (default 50).

Exit 0 when every clip meets its criterion; exit 1 otherwise.

score_rep() API (forward declaration)
--------------------------------------
    score_rep(frames: list[np.ndarray], technique: str) -> dict

    frames    : list of N arrays, each shape (33, 3) — MediaPipe Pose
                landmark coordinates in the normalised hip-centred space
                described in the README (Y-up, Z-forward, torso scale = 1).
    technique : one of "front_kick", "roundhouse_kick", "straight_punch".

    Returns a dict with keys:
        "overall"   : float, weighted mean of criterion scores (0–100)
        "criteria"  : dict[str, float], per-criterion scores (0–100)

Adding real clip sequences
--------------------------
1. Extract MediaPipe landmarks from a recorded video and normalise them
   using the project's vision module (once implemented):

       from kime.vision import extract_landmarks, normalize_sequence
       raw    = extract_landmarks("path/to/clip.mp4")    # list[np.ndarray (33,3)]
       frames = normalize_sequence(raw)                   # hip-centred, scaled

2. Append a ClipSpec to CLIPS inside _build_clips() near the bottom of
   this file:

       ClipSpec(
           technique="roundhouse_kick",
           label="expert",          # "expert" or "bad"
           frames=frames,
       )

3. Re-run `python scripts/sanity_eval.py` to see the updated table.

The sequences bundled here are *synthetic*: they are analytically constructed
poses that satisfy (or deliberately violate) each biomechanical rule, so no
real video is needed to verify the scoring logic.  Once real reference clips
are available from the data-scraping phase, swap them in via step 2 above.

Coordinate system (after normalisation)
-----------------------------------------
    Origin : hip centre
    Y-axis : up  (head at approx (0, 1.7, 0))
    X-axis : right (positive = practitioner's right)
    Z-axis : forward (positive = toward camera)
    Scale  : torso length (hip-centre → mid-shoulder) normalised to 1.0
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (33 total; only the subset used is named)
# ---------------------------------------------------------------------------
NOSE        = 0
L_SHOULDER  = 11
R_SHOULDER  = 12
L_ELBOW     = 13
R_ELBOW     = 14
L_WRIST     = 15
R_WRIST     = 16
L_HIP       = 23
R_HIP       = 24
L_KNEE      = 25
R_KNEE      = 26
L_ANKLE     = 27
R_ANKLE     = 28
L_HEEL      = 29
R_HEEL      = 30

N_LANDMARKS = 33

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle in degrees at vertex *b* formed by the points a–b–c."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos_val = float(np.dot(v1, v2) / denom)
    return math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))


def _hip_rotation_deg(frame: np.ndarray) -> float:
    """
    Return the hip-line rotation angle (degrees) measured in the XZ plane
    (i.e. around the vertical Y-axis).  Positive = right hip forward (+Z).
    Zero means hips are square to the camera (neutral stance baseline).
    """
    hip_vec = frame[R_HIP] - frame[L_HIP]
    return math.degrees(math.atan2(float(hip_vec[2]), float(hip_vec[0])))


def _shoulder_rotation_deg(frame: np.ndarray) -> float:
    """Same as _hip_rotation_deg but for the shoulder girdle."""
    sh_vec = frame[R_SHOULDER] - frame[L_SHOULDER]
    return math.degrees(math.atan2(float(sh_vec[2]), float(sh_vec[0])))


# ---------------------------------------------------------------------------
# Neutral standing pose  (33 × 3, hip-centred, Y-up, Z-forward)
# ---------------------------------------------------------------------------


def _neutral_pose() -> np.ndarray:
    """Return (33, 3) array for a relaxed guard stance."""
    p = np.zeros((N_LANDMARKS, 3))

    # Head
    p[NOSE]        = [ 0.00,  1.75,  0.05]
    # Shoulders
    p[L_SHOULDER]  = [-0.30,  1.20,  0.00]
    p[R_SHOULDER]  = [ 0.30,  1.20,  0.00]
    # Elbows (arms bent, guard position)
    p[L_ELBOW]     = [-0.40,  0.75,  0.05]
    p[R_ELBOW]     = [ 0.40,  0.75,  0.05]
    # Wrists (hands up near jaw — guard)
    p[L_WRIST]     = [-0.28,  1.10,  0.12]
    p[R_WRIST]     = [ 0.28,  1.10,  0.12]
    # Hips
    p[L_HIP]       = [-0.15,  0.00,  0.00]
    p[R_HIP]       = [ 0.15,  0.00,  0.00]
    # Knees
    p[L_KNEE]      = [-0.15, -0.50,  0.00]
    p[R_KNEE]      = [ 0.15, -0.50,  0.00]
    # Ankles
    p[L_ANKLE]     = [-0.15, -1.00,  0.00]
    p[R_ANKLE]     = [ 0.15, -1.00,  0.00]
    # Heels
    p[L_HEEL]      = [-0.15, -1.05, -0.06]
    p[R_HEEL]      = [ 0.15, -1.05, -0.06]

    return p


# ---------------------------------------------------------------------------
# Sequence construction helpers
# ---------------------------------------------------------------------------


def _lerp_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _build_sequence(
    keyframes: List[np.ndarray], steps_between: int = 4
) -> List[np.ndarray]:
    """
    Linearly interpolate *steps_between* frames between each consecutive pair
    of key frames and return the full flat list (first key to last key,
    inclusive).
    """
    out: List[np.ndarray] = [keyframes[0].copy()]
    for i in range(len(keyframes) - 1):
        for s in range(1, steps_between + 1):
            t = s / steps_between
            out.append(_lerp_pose(keyframes[i], keyframes[i + 1], t))
    return out


# ---------------------------------------------------------------------------
# Synthetic clip generators — FRONT KICK
# ---------------------------------------------------------------------------


def _front_kick_expert_clip() -> List[np.ndarray]:
    """
    Expert right front kick.

    Biomechanical intent
    --------------------
    * Knee chambers well above hip (R_KNEE_y − R_HIP_y ≈ +0.25).
    * Leg extends to ~162° at impact (near full extension).
      Verified: hip=(0.15,0,0), knee=(0.15,0.25,0.43), ankle=(0.15,0.35,0.92)
                v1=(0,−0.25,−0.43) v2=(0,0.10,0.49)  cosθ≈−0.951 → θ≈162°
    * Guard hand (L_WRIST) stays high throughout.
    * Hip centre remains close to support (left) ankle — good balance.
    * Hips stay level (tiny tilt of 0.02).
    """
    neutral = _neutral_pose()

    # Chamber: right knee rises to chest height
    chamber = neutral.copy()
    chamber[R_KNEE]  = np.array([ 0.15,  0.25,  0.10])
    chamber[R_ANKLE] = np.array([ 0.18,  0.00,  0.18])

    # Impact: near full extension
    impact = neutral.copy()
    impact[L_HIP]    = np.array([-0.17,  0.00,  0.00])
    impact[R_HIP]    = np.array([ 0.12,  0.02,  0.00])  # tiny tilt
    impact[R_KNEE]   = np.array([ 0.15,  0.25,  0.43])
    impact[R_ANKLE]  = np.array([ 0.15,  0.35,  0.92])
    impact[L_WRIST]  = np.array([-0.28,  1.10,  0.12])  # guard stays up
    impact[R_WRIST]  = np.array([ 0.28,  0.80,  0.20])  # rear hand drops naturally

    return _build_sequence([neutral, neutral, chamber, impact, chamber, neutral])


def _front_kick_bad_clip() -> List[np.ndarray]:
    """
    Deliberately poor right front kick.

    Deficiencies
    ------------
    * Low chamber: knee barely rises (R_KNEE_y − R_HIP_y ≈ −0.10 still below hip).
    * Misdirected knee drifts outward — poor extension angle (~75°, below straight-line).
      Verified: hip=(0.15,0,0), knee=(0.30,0.10,0.10), ankle=(0.35,−0.40,0.25)
                v1=(−0.15,−0.10,−0.10) v2=(0.05,−0.50,0.15) cosθ≈+0.255 → θ≈75°
    * Guard drops to waist level.
    * Excessive backward lean — hip centre well left of left ankle.
    * Right hip drops 0.15 below left (significant tilt).
    """
    neutral = _neutral_pose()

    # Chamber: knee barely rises, still below hip level
    chamber = neutral.copy()
    chamber[R_KNEE]  = np.array([ 0.22, -0.10,  0.05])
    chamber[R_ANKLE] = np.array([ 0.24, -0.45,  0.08])

    # Impact: knee drifts sideways instead of driving forward; leg badly extended
    impact = neutral.copy()
    impact[L_HIP]    = np.array([-0.42,  0.00,  0.00])   # excessive backward lean
    impact[R_HIP]    = np.array([-0.22, -0.15,  0.00])   # right hip drops 0.15
    impact[R_KNEE]   = np.array([ 0.30,  0.10,  0.10])   # knee drifts out and barely up
    impact[R_ANKLE]  = np.array([ 0.35, -0.40,  0.25])   # ankle flares to the side
    impact[L_WRIST]  = np.array([-0.28,  0.20,  0.05])   # guard dropped to waist
    impact[R_WRIST]  = np.array([ 0.28,  0.20,  0.05])

    return _build_sequence([neutral, neutral, chamber, impact, chamber, neutral])


# ---------------------------------------------------------------------------
# Synthetic clip generators — ROUNDHOUSE KICK
# ---------------------------------------------------------------------------


def _roundhouse_kick_expert_clip() -> List[np.ndarray]:
    """
    Expert right roundhouse kick.

    Biomechanical intent
    --------------------
    * Knee chambers high to the side (R_KNEE_y − R_HIP_y ≈ +0.20).
    * Hips rotate ~45° at impact (right hip drives forward).
      hip_vec = R_HIP − L_HIP = (0.20, 0, 0.20) → atan2(0.20,0.20) = 45°
    * Leg extends to ~151° at impact.
      Verified: hip=(0.10,0,0.10), knee=(0.20,0.15,0.50), ankle=(0.25,0.10,0.85)
                v1=(−0.10,−0.15,−0.40) v2=(0.05,−0.05,0.35) cosθ≈−0.877 → θ≈151°
    * Guard maintained.
    """
    neutral = _neutral_pose()

    chamber = neutral.copy()
    chamber[R_KNEE]  = np.array([ 0.45,  0.30,  0.05])  # knee high and out (above hip)
    chamber[R_ANKLE] = np.array([ 0.38, -0.05,  0.00])

    impact = neutral.copy()
    impact[L_HIP]    = np.array([-0.10,  0.00, -0.10])
    impact[R_HIP]    = np.array([ 0.10,  0.02,  0.10])
    impact[R_KNEE]   = np.array([ 0.20,  0.15,  0.50])
    impact[R_ANKLE]  = np.array([ 0.25,  0.10,  0.85])
    impact[L_WRIST]  = np.array([-0.28,  1.10,  0.12])
    impact[R_WRIST]  = np.array([ 0.28,  0.80,  0.15])

    return _build_sequence([neutral, neutral, chamber, impact, chamber, neutral])


def _roundhouse_kick_bad_clip() -> List[np.ndarray]:
    """
    Deliberately poor right roundhouse kick — zero hip rotation.

    The most common roundhouse fault: the leg swings without rotating the
    hips through the kick, drastically reducing power.  This causes
    hip_rotation to score < 10.

    Additional deficiencies
    -----------------------
    * Low chamber.
    * Incomplete extension (~118°).
      Verified: hip=(0.15,0,0), knee=(0.22,0,0.18), ankle=(0.28,−0.25,0.30)
                v1=(−0.07,0,−0.18) v2=(0.06,−0.25,0.12) cosθ≈−0.471 → θ≈118°
    * Guard drops.
    """
    neutral = _neutral_pose()

    chamber = neutral.copy()
    chamber[R_KNEE]  = np.array([ 0.30, -0.10,  0.05])  # barely raised
    chamber[R_ANKLE] = np.array([ 0.28, -0.40,  0.02])

    impact = neutral.copy()
    impact[L_HIP]    = np.array([-0.15,  0.00, -0.01])   # essentially no rotation
    impact[R_HIP]    = np.array([ 0.15, -0.15,  0.01])   # hip drops
    impact[R_KNEE]   = np.array([ 0.22,  0.00,  0.18])
    impact[R_ANKLE]  = np.array([ 0.28, -0.25,  0.30])
    impact[L_WRIST]  = np.array([-0.28,  0.25,  0.05])   # guard dropped
    impact[R_WRIST]  = np.array([ 0.28,  0.25,  0.05])

    return _build_sequence([neutral, neutral, chamber, impact, chamber, neutral])


# ---------------------------------------------------------------------------
# Synthetic clip generators — STRAIGHT PUNCH
# ---------------------------------------------------------------------------


def _straight_punch_expert_clip() -> List[np.ndarray]:
    """
    Expert right straight punch.

    Biomechanical intent
    --------------------
    * Arm fully extends — elbow angle ≈ 175° at impact.
      Verified: shoulder=(0.25,1.15,0), elbow=(0.28,1.10,0.35), wrist=(0.30,1.05,0.70)
                v1=(−0.03,0.05,−0.35) v2=(0.02,−0.05,0.35) cosθ≈−0.996 → θ≈175°
    * Hips rotate ~18° into the punch (right hip drives forward).
    * Shoulders rotate ~22° in concert with hips.
    * Guard (left) wrist stays high throughout.
    * Wrist extends well forward (Z ≈ 0.70).
    """
    neutral = _neutral_pose()

    windup = neutral.copy()
    windup[R_ELBOW]  = np.array([ 0.48,  0.70, -0.10])
    windup[R_WRIST]  = np.array([ 0.38,  1.05, -0.05])

    impact = neutral.copy()
    impact[L_HIP]      = np.array([-0.15,  0.00, -0.05])
    impact[R_HIP]      = np.array([ 0.15,  0.00,  0.05])
    impact[L_SHOULDER] = np.array([-0.30,  1.20, -0.07])
    impact[R_SHOULDER] = np.array([ 0.25,  1.15,  0.00])
    impact[R_ELBOW]    = np.array([ 0.28,  1.10,  0.35])
    impact[R_WRIST]    = np.array([ 0.30,  1.05,  0.70])
    impact[L_WRIST]    = np.array([-0.28,  1.10,  0.12])  # guard stays up

    retract = impact.copy()
    retract[R_ELBOW]   = np.array([ 0.40,  0.75,  0.15])
    retract[R_WRIST]   = np.array([ 0.28,  1.05,  0.15])

    return _build_sequence([neutral, windup, impact, retract, neutral])


def _straight_punch_bad_clip() -> List[np.ndarray]:
    """
    Deliberately poor right straight punch.

    Deficiencies
    ------------
    * Arm barely extends — elbow angle ≈ 96° (no drive through the target).
      Verified: shoulder=(0.30,1.20,0), elbow=(0.50,1.00,0.15), wrist=(0.40,0.95,0.25)
                v1=(−0.20,0.20,−0.15) v2=(−0.10,−0.05,0.10) cosθ≈−0.104 → θ≈96°
    * No hip rotation (hips stay square).
    * No shoulder rotation.
    * Guard drops to waist level.
    * Wrist barely moves forward (Z ≈ 0.25).
    """
    neutral = _neutral_pose()

    windup = neutral.copy()
    windup[R_ELBOW]  = np.array([ 0.48,  0.70, -0.10])
    windup[R_WRIST]  = np.array([ 0.38,  1.05, -0.05])

    impact = neutral.copy()
    # Hips and shoulders stay square
    impact[L_HIP]      = np.array([-0.15,  0.00,  0.00])
    impact[R_HIP]      = np.array([ 0.15,  0.00,  0.00])
    impact[L_SHOULDER] = np.array([-0.30,  1.20,  0.00])
    impact[R_SHOULDER] = np.array([ 0.30,  1.20,  0.00])
    # Elbow flares out, arm does not extend
    impact[R_ELBOW]    = np.array([ 0.50,  1.00,  0.15])
    impact[R_WRIST]    = np.array([ 0.40,  0.95,  0.25])
    # Guard dropped
    impact[L_WRIST]    = np.array([-0.28,  0.25,  0.05])

    retract = impact.copy()
    retract[R_ELBOW]   = np.array([ 0.40,  0.75,  0.05])
    retract[R_WRIST]   = np.array([ 0.28,  1.05,  0.05])

    return _build_sequence([neutral, windup, impact, retract, neutral])


# ---------------------------------------------------------------------------
# Scoring criteria (rule-based biomechanics)
# ---------------------------------------------------------------------------


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


# ---- Front kick -------------------------------------------------------


def _score_front_kick(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Criteria
    --------
    chamber_height  : max (R_KNEE_y − R_HIP_y) across all frames, anchored to
                      the first-frame hip height so a dropped hip during kick
                      doesn't inflate the metric.
                      Scale: 0.0 (knee at hip level) → 0,  +0.30 → 100.
    extension_angle : angle at R_KNEE (R_HIP–R_KNEE–R_ANKLE) at the impact
                      frame (highest R_ANKLE_y).
                      Scale: 90° → 0,  170° → 100.
    guard_position  : L_WRIST_y / (shoulder_mid_y + 0.1) × 100 at impact.
    balance         : 100 − |hip_centre_x − L_ANKLE_x| × 300 at impact.
    hip_alignment   : 100 − |L_HIP_y − R_HIP_y| × 400 at impact (level hips).
    """
    impact_idx = int(np.argmax([f[R_ANKLE][1] for f in frames]))
    imp = frames[impact_idx]

    # 1. Chamber height — use the first frame's hip Y as a stable reference so
    #    a hip that drops during the kick doesn't artificially inflate the score.
    ref_hip_y = float(frames[0][R_HIP][1])
    max_chamber = max(float(f[R_KNEE][1]) - ref_hip_y for f in frames)
    chamber_score = _clamp(max_chamber / 0.30 * 100.0)

    # 2. Extension angle
    ext_angle = _angle_3pt(imp[R_HIP], imp[R_KNEE], imp[R_ANKLE])
    ext_score = _clamp((ext_angle - 90.0) / 80.0 * 100.0)

    # 3. Guard position
    shoulder_mid_y = float((imp[L_SHOULDER][1] + imp[R_SHOULDER][1]) / 2.0)
    guard_score = _clamp(float(imp[L_WRIST][1]) / (shoulder_mid_y + 0.1) * 100.0)

    # 4. Balance
    hip_centre_x = float((imp[L_HIP][0] + imp[R_HIP][0]) / 2.0)
    balance_err = abs(hip_centre_x - float(imp[L_ANKLE][0]))
    balance_score = _clamp(100.0 - balance_err * 300.0)

    # 5. Hip alignment (level hips)
    tilt = abs(float(imp[L_HIP][1] - imp[R_HIP][1]))
    hip_align_score = _clamp(100.0 - tilt * 400.0)

    criteria = {
        "chamber_height":  round(chamber_score,  1),
        "extension_angle": round(ext_score,       1),
        "guard_position":  round(guard_score,     1),
        "balance":         round(balance_score,   1),
        "hip_alignment":   round(hip_align_score, 1),
    }
    overall = round(sum(criteria.values()) / len(criteria), 1)
    return {"overall": overall, "criteria": criteria}


# ---- Roundhouse kick --------------------------------------------------


def _score_roundhouse_kick(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Criteria
    --------
    chamber_height  : same formula as front kick (stable first-frame hip reference).
    hip_rotation    : hip-line XZ rotation angle (degrees) at impact frame.
                      Scale: 0° → 0,  40° → 100.
    extension_angle : angle at R_KNEE at impact frame.
    guard_position  : same formula as front kick.
    balance         : same formula as front kick.
    """
    impact_idx = int(np.argmax([f[R_ANKLE][1] for f in frames]))
    imp = frames[impact_idx]

    # 1. Chamber height — stable first-frame hip reference (see front kick scorer)
    ref_hip_y = float(frames[0][R_HIP][1])
    max_chamber = max(float(f[R_KNEE][1]) - ref_hip_y for f in frames)
    chamber_score = _clamp(max_chamber / 0.30 * 100.0)

    # 2. Hip rotation — primary roundhouse criterion
    rot_deg = _hip_rotation_deg(imp)
    hip_rot_score = _clamp(rot_deg / 40.0 * 100.0)

    # 3. Extension angle
    ext_angle = _angle_3pt(imp[R_HIP], imp[R_KNEE], imp[R_ANKLE])
    ext_score = _clamp((ext_angle - 90.0) / 80.0 * 100.0)

    # 4. Guard position
    shoulder_mid_y = float((imp[L_SHOULDER][1] + imp[R_SHOULDER][1]) / 2.0)
    guard_score = _clamp(float(imp[L_WRIST][1]) / (shoulder_mid_y + 0.1) * 100.0)

    # 5. Balance
    hip_centre_x = float((imp[L_HIP][0] + imp[R_HIP][0]) / 2.0)
    balance_err = abs(hip_centre_x - float(imp[L_ANKLE][0]))
    balance_score = _clamp(100.0 - balance_err * 300.0)

    criteria = {
        "chamber_height":  round(chamber_score,  1),
        "hip_rotation":    round(hip_rot_score,  1),
        "extension_angle": round(ext_score,      1),
        "guard_position":  round(guard_score,    1),
        "balance":         round(balance_score,  1),
    }
    overall = round(sum(criteria.values()) / len(criteria), 1)
    return {"overall": overall, "criteria": criteria}


# ---- Straight punch ---------------------------------------------------


def _score_straight_punch(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Criteria
    --------
    extension_angle  : angle at R_ELBOW (R_SHOULDER–R_ELBOW–R_WRIST) at
                       impact frame (highest R_WRIST_z).
                       Scale: 90° → 0,  170° → 100.
    hip_rotation     : hip-line XZ rotation at impact.
                       Scale: 0° → 0,  25° → 100.
    guard_position   : L_WRIST_y / (shoulder_mid_y + 0.1) × 100 at impact.
    shoulder_rotation: shoulder-line XZ rotation at impact.
                       Scale: 0° → 0,  25° → 100.
    extension_depth  : R_WRIST_z at impact. Scale: 0.3 → 0,  0.7 → 100.
    """
    impact_idx = int(np.argmax([f[R_WRIST][2] for f in frames]))
    imp = frames[impact_idx]

    # 1. Extension angle
    ext_angle = _angle_3pt(imp[R_SHOULDER], imp[R_ELBOW], imp[R_WRIST])
    ext_score = _clamp((ext_angle - 90.0) / 80.0 * 100.0)

    # 2. Hip rotation
    rot_deg = _hip_rotation_deg(imp)
    hip_rot_score = _clamp(rot_deg / 25.0 * 100.0)

    # 3. Guard position
    shoulder_mid_y = float((imp[L_SHOULDER][1] + imp[R_SHOULDER][1]) / 2.0)
    guard_score = _clamp(float(imp[L_WRIST][1]) / (shoulder_mid_y + 0.1) * 100.0)

    # 4. Shoulder rotation
    sh_rot_deg = _shoulder_rotation_deg(imp)
    sh_rot_score = _clamp(sh_rot_deg / 25.0 * 100.0)

    # 5. Extension depth
    depth_score = _clamp((float(imp[R_WRIST][2]) - 0.3) / 0.4 * 100.0)

    criteria = {
        "extension_angle":   round(ext_score,     1),
        "hip_rotation":      round(hip_rot_score,  1),
        "guard_position":    round(guard_score,    1),
        "shoulder_rotation": round(sh_rot_score,   1),
        "extension_depth":   round(depth_score,    1),
    }
    overall = round(sum(criteria.values()) / len(criteria), 1)
    return {"overall": overall, "criteria": criteria}


# ---------------------------------------------------------------------------
# Public scoring entry point
# ---------------------------------------------------------------------------

TECHNIQUE_SCORERS = {
    "front_kick":      _score_front_kick,
    "roundhouse_kick": _score_roundhouse_kick,
    "straight_punch":  _score_straight_punch,
}


def score_rep(frames: List[np.ndarray], technique: str) -> Dict:
    """
    Score a single rep given its landmark sequence and technique name.

    Parameters
    ----------
    frames    : list of np.ndarray, each shape (33, 3).
                MediaPipe Pose landmarks in normalised hip-centred coordinates.
    technique : ``"front_kick"`` | ``"roundhouse_kick"`` | ``"straight_punch"``

    Returns
    -------
    dict with keys:
        ``"overall"``   : float  — mean criterion score 0–100
        ``"criteria"``  : dict[str, float] — per-criterion scores 0–100
    """
    if technique not in TECHNIQUE_SCORERS:
        raise ValueError(
            f"Unknown technique {technique!r}. "
            f"Valid options: {sorted(TECHNIQUE_SCORERS)}"
        )
    if not frames:
        raise ValueError("frames must be a non-empty list")
    return TECHNIQUE_SCORERS[technique](frames)


# ---------------------------------------------------------------------------
# Clip registry
# ---------------------------------------------------------------------------


@dataclass
class ClipSpec:
    technique: str
    label: str                           # "expert" or "bad"
    frames: List[np.ndarray] = field(repr=False)


def _build_clips() -> List[ClipSpec]:
    """Build and return the full list of clips to evaluate."""
    return [
        # ---- Front kick ----------------------------------------
        ClipSpec("front_kick",      "expert", _front_kick_expert_clip()),
        ClipSpec("front_kick",      "bad",    _front_kick_bad_clip()),
        # ---- Roundhouse kick -----------------------------------
        ClipSpec("roundhouse_kick", "expert", _roundhouse_kick_expert_clip()),
        ClipSpec("roundhouse_kick", "bad",    _roundhouse_kick_bad_clip()),
        # ---- Straight punch ------------------------------------
        ClipSpec("straight_punch",  "expert", _straight_punch_expert_clip()),
        ClipSpec("straight_punch",  "bad",    _straight_punch_bad_clip()),
    ]


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _collect_all_criteria(results: List[Tuple[ClipSpec, Dict]]) -> List[str]:
    """Return stable deduplicated criterion names across all results."""
    seen: List[str] = []
    for _, r in results:
        for k in r["criteria"]:
            if k not in seen:
                seen.append(k)
    return seen


def _print_table(
    results: List[Tuple[ClipSpec, Dict]],
    expert_threshold: float,
    bad_ceiling: float,
) -> bool:
    """
    Print the evaluation table.

    Returns True if every clip meets its criterion (expert >= threshold,
    bad <= ceiling), False otherwise.
    """
    all_criteria = _collect_all_criteria(results)

    # Column widths — criterion columns sized to fit the longest name exactly
    W_TECHNIQUE = 18
    W_LABEL     = 6
    W_OVERALL   = 7
    W_CRITERION = max((len(c) for c in all_criteria), default=10)
    W_RESULT    = 10

    header = (
        f"{'Technique':<{W_TECHNIQUE}}  "
        f"{'Label':<{W_LABEL}}  "
        f"{'Overall':>{W_OVERALL}}"
    )
    for crit in all_criteria:
        header += f"  {crit:>{W_CRITERION}}"
    header += f"  {'Result':>{W_RESULT}}"

    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_pass = True
    for clip, result in results:
        overall = result["overall"]
        crit_dict = result["criteria"]

        if clip.label == "expert":
            passes = overall >= expert_threshold
            target = f">={expert_threshold:.0f}"
        else:
            passes = overall <= bad_ceiling
            target = f"<={bad_ceiling:.0f}"

        if not passes:
            all_pass = False

        status = "PASS" if passes else f"FAIL({target})"

        row = (
            f"{clip.technique:<{W_TECHNIQUE}}  "
            f"{clip.label:<{W_LABEL}}  "
            f"{overall:>{W_OVERALL}.1f}"
        )
        for crit in all_criteria:
            val = crit_dict.get(crit)
            cell = f"{val:>{W_CRITERION}.1f}" if val is not None else f"{'—':>{W_CRITERION}}"
            row += f"  {cell}"
        row += f"  {status:>{W_RESULT}}"
        print(row)

    print(sep)
    print(
        f"\nThresholds:  expert >= {expert_threshold:.0f}   |   bad <= {bad_ceiling:.0f}"
    )
    verdict = "ALL PASS" if all_pass else "SOME FAILURES — see FAIL rows above"
    print(f"Result: {verdict}")
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kime sanity eval: run synthetic expert/bad clips through score_rep() "
            "and verify that scores meet their expected thresholds."
        )
    )
    parser.add_argument(
        "--expert-threshold",
        type=float,
        default=75.0,
        metavar="N",
        help="Minimum score required for expert clips (default: 75)",
    )
    parser.add_argument(
        "--bad-ceiling",
        type=float,
        default=50.0,
        metavar="N",
        help="Maximum score allowed for bad clips (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame impact pose details for each clip",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    clips = _build_clips()
    results: List[Tuple[ClipSpec, Dict]] = []

    for clip in clips:
        result = score_rep(clip.frames, clip.technique)
        results.append((clip, result))

        if args.verbose:
            print(
                f"[verbose] {clip.technique}/{clip.label}: "
                f"{len(clip.frames)} frames  overall={result['overall']}"
            )
            for k, v in result["criteria"].items():
                print(f"           {k} = {v}")

    print(f"\nKime Sanity Eval — {len(clips)} clips ({len(clips) // 2} techniques)\n")
    all_pass = _print_table(results, args.expert_threshold, args.bad_ceiling)
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
