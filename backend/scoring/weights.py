"""
Per-technique criterion weights for the Kime scoring engine.

Each dict maps criterion name -> float weight in [0, 1].
Weights within a technique must sum to 1.0.

The six criteria correspond to the biomechanical checkpoints described in the
judging pipeline:
  chamber_height   — how high the knee/hand is raised during the chamber phase
  hip_rotation     — degree of hip rotation at the moment of impact
  extension_angle  — joint angle at full extension (knee for kicks, elbow for punch)
  balance          — centre-of-mass proximity to the support base at impact
  guard_position   — non-striking hand/arm protecting the centreline
  retraction_speed — how quickly the limb returns after impact

Weights are tuned per technique because, e.g., hip rotation matters far more for
a roundhouse kick than for a straight punch, while extension angle is critical for
a punch but secondary for a front kick.
"""

TECHNIQUE_WEIGHTS: dict[str, dict[str, float]] = {
    "front_kick": {
        "chamber_height": 0.20,
        "hip_rotation": 0.10,
        "extension_angle": 0.25,
        "balance": 0.20,
        "guard_position": 0.15,
        "retraction_speed": 0.10,
    },
    "roundhouse_kick": {
        "chamber_height": 0.15,
        "hip_rotation": 0.25,
        "extension_angle": 0.20,
        "balance": 0.15,
        "guard_position": 0.10,
        "retraction_speed": 0.15,
    },
    "straight_punch": {
        "chamber_height": 0.05,
        "hip_rotation": 0.20,
        "extension_angle": 0.30,
        "balance": 0.20,
        "guard_position": 0.15,
        "retraction_speed": 0.10,
    },
}

# Ordered list of criterion names — the engine uses this to guarantee a
# consistent iteration order and to enforce exactly-six entries.
CRITERION_NAMES: tuple[str, ...] = (
    "chamber_height",
    "hip_rotation",
    "extension_angle",
    "balance",
    "guard_position",
    "retraction_speed",
)

# Validate weights at import time so a mis-edit is caught immediately.
_TOLERANCE = 1e-6
for _technique, _weights in TECHNIQUE_WEIGHTS.items():
    _missing = set(CRITERION_NAMES) - set(_weights)
    _extra = set(_weights) - set(CRITERION_NAMES)
    if _missing:
        raise ValueError(
            f"weights.py: {_technique!r} is missing criteria: {_missing}"
        )
    if _extra:
        raise ValueError(
            f"weights.py: {_technique!r} has unknown criteria: {_extra}"
        )
    _total = sum(_weights.values())
    if abs(_total - 1.0) > _TOLERANCE:
        raise ValueError(
            f"weights.py: {_technique!r} weights sum to {_total:.6f}, expected 1.0"
        )
