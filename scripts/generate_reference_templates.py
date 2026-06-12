"""Generate synthetic reference templates for development use.

Produces one ``.npy`` file per technique under ``backend/data/references/``.
Each file is a ``(60, 99)`` float64 array representing a 60-frame normalized
pose sequence for an idealized technique repetition.

**Coordinate convention** (matches the normalization pipeline output):
  - Hip midpoint at origin (hip-centered, torso-scaled)
  - y increases *downward* (image-space: y > 0 below hips, y < 0 above hips)
  - x positive = right, z positive = toward camera
  - Torso length (hip midpoint → shoulder midpoint) ≈ 1.0

Landmark indices follow the MediaPipe Pose 33-point model.

Usage::

    python scripts/generate_reference_templates.py

These templates are intentionally schematic.  They establish a plausible
reference for development, CI, and the sanity-eval script.  Once the Phase 2
data pipeline is run against real expert footage and human-reviewed, the
resulting templates should replace these files.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "backend" / "data" / "references"

_N_FRAMES = 60
_N_LANDMARKS = 33
_N_FEATURES = _N_LANDMARKS * 3  # 99


# ---------------------------------------------------------------------------
# Base skeleton (T-pose standing, hip-centered, unit-torso scale)
# ---------------------------------------------------------------------------

_BASE = np.zeros((_N_LANDMARKS, 3), dtype=np.float64)

# Head / face cluster
_BASE[0]  = [ 0.00, -1.80,  0.00]  # nose
_BASE[1]  = [-0.05, -1.85,  0.05]  # left eye inner
_BASE[2]  = [-0.08, -1.85,  0.05]  # left eye
_BASE[3]  = [-0.11, -1.85,  0.05]  # left eye outer
_BASE[4]  = [ 0.05, -1.85,  0.05]  # right eye inner
_BASE[5]  = [ 0.08, -1.85,  0.05]  # right eye
_BASE[6]  = [ 0.11, -1.85,  0.05]  # right eye outer
_BASE[7]  = [-0.15, -1.82,  0.00]  # left ear
_BASE[8]  = [ 0.15, -1.82,  0.00]  # right ear
_BASE[9]  = [-0.03, -1.75,  0.05]  # mouth left
_BASE[10] = [ 0.03, -1.75,  0.05]  # mouth right

# Shoulders
_BASE[11] = [-0.25, -1.50,  0.00]  # left shoulder
_BASE[12] = [ 0.25, -1.50,  0.00]  # right shoulder

# Elbows — arms in guard: bent upward toward chin
_BASE[13] = [-0.30, -1.20,  0.10]  # left elbow
_BASE[14] = [ 0.30, -1.20,  0.10]  # right elbow

# Wrists — guard position near chin
_BASE[15] = [-0.20, -1.60,  0.20]  # left wrist
_BASE[16] = [ 0.20, -1.60,  0.20]  # right wrist

# Fingers / thumbs (approximate guard position)
_BASE[17] = [-0.22, -1.65,  0.22]  # left pinky
_BASE[18] = [ 0.22, -1.65,  0.22]  # right pinky
_BASE[19] = [-0.19, -1.65,  0.22]  # left index
_BASE[20] = [ 0.19, -1.65,  0.22]  # right index
_BASE[21] = [-0.20, -1.63,  0.21]  # left thumb
_BASE[22] = [ 0.20, -1.63,  0.21]  # right thumb

# Hips (at origin after hip-centering)
_BASE[23] = [-0.12,  0.00,  0.00]  # left hip
_BASE[24] = [ 0.12,  0.00,  0.00]  # right hip

# Knees
_BASE[25] = [-0.12,  0.50,  0.00]  # left knee
_BASE[26] = [ 0.12,  0.50,  0.00]  # right knee

# Ankles
_BASE[27] = [-0.12,  1.00,  0.00]  # left ankle
_BASE[28] = [ 0.12,  1.00,  0.00]  # right ankle

# Heels
_BASE[29] = [-0.13,  1.05, -0.05]  # left heel
_BASE[30] = [ 0.13,  1.05, -0.05]  # right heel

# Foot index tips
_BASE[31] = [-0.10,  1.05,  0.10]  # left foot index
_BASE[32] = [ 0.10,  1.05,  0.10]  # right foot index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _build_sequence(keyframes: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Linearly interpolate between keyframes to produce a (N_FRAMES, 33, 3) array."""
    seq = np.zeros((_N_FRAMES, _N_LANDMARKS, 3), dtype=np.float64)
    for seg in range(len(keyframes) - 1):
        f0, pose0 = keyframes[seg]
        f1, pose1 = keyframes[seg + 1]
        for f in range(f0, f1 + 1):
            t = (f - f0) / max(f1 - f0, 1)
            seq[f] = _lerp(pose0, pose1, t)
    return seq


def _add_noise(seq: np.ndarray, sigma: float = 0.005, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return seq + rng.normal(0.0, sigma, seq.shape)


# ---------------------------------------------------------------------------
# Front kick (mae geri) — right leg
# Phases: guard (0-9) → chamber knee high (10-24) → full extension (25-44)
#         → retract to guard (45-59)
# ---------------------------------------------------------------------------


def _make_front_kick() -> np.ndarray:
    guard = _BASE.copy()

    chamber = _BASE.copy()
    # Right knee rises above hip (y goes negative = upward)
    chamber[26] = [ 0.15, -0.10,  0.10]   # right knee raised
    chamber[28] = [ 0.18,  0.15,  0.20]   # right ankle tucked
    chamber[30] = [ 0.19,  0.20,  0.15]   # right heel
    chamber[32] = [ 0.17,  0.20,  0.25]   # right foot index

    extension = _BASE.copy()
    # Right leg drives forward to full extension
    extension[26] = [ 0.15,  0.30,  0.40]  # right knee extends
    extension[28] = [ 0.16,  0.35,  0.80]  # right ankle at target
    extension[30] = [ 0.17,  0.40,  0.75]  # right heel
    extension[32] = [ 0.15,  0.30,  0.90]  # right foot index (tip)

    seq = _build_sequence([
        (0,  guard),
        (9,  guard),
        (24, chamber),
        (44, extension),
        (59, guard),
    ])
    return _add_noise(seq, seed=42)


# ---------------------------------------------------------------------------
# Roundhouse kick (mawashi geri) — right leg
# Phases: guard (0-9) → chamber-to-side (10-24) → hip-rotation+sweep (25-44)
#         → retract to guard (45-59)
# ---------------------------------------------------------------------------


def _make_roundhouse_kick() -> np.ndarray:
    guard = _BASE.copy()

    chamber = _BASE.copy()
    # Right knee raised out to the side; hip begins rotating
    chamber[26] = [ 0.30, -0.05,  0.05]   # right knee up and out
    chamber[28] = [ 0.45,  0.20,  0.05]   # right ankle tucked in
    chamber[30] = [ 0.46,  0.25,  0.00]
    chamber[32] = [ 0.46,  0.20,  0.10]
    chamber[24] = [ 0.12,  0.00, -0.08]   # right hip starts forward rotation

    extension = _BASE.copy()
    # Maximum hip yaw; right leg sweeps horizontally through target
    extension[23] = [-0.05,  0.00, -0.10]  # left hip rotates back
    extension[24] = [ 0.15,  0.00,  0.15]  # right hip rotates forward
    extension[26] = [ 0.50,  0.20,  0.00]  # right knee out
    extension[28] = [ 0.70,  0.25, -0.30]  # right ankle sweeps through
    extension[30] = [ 0.68,  0.28, -0.35]
    extension[32] = [ 0.72,  0.20, -0.25]
    # Upper body counter-rotates slightly for balance
    extension[11] = [-0.22, -1.50,  0.08]
    extension[12] = [ 0.28, -1.50, -0.05]

    seq = _build_sequence([
        (0,  guard),
        (9,  guard),
        (24, chamber),
        (44, extension),
        (59, guard),
    ])
    return _add_noise(seq, seed=7)


# ---------------------------------------------------------------------------
# Straight punch (choku tsuki / jab) — right hand
# Phases: guard (0-9) → full extension (10-29) → retract (30-44)
#         → guard (45-59)
# ---------------------------------------------------------------------------


def _make_straight_punch() -> np.ndarray:
    guard = _BASE.copy()

    extension = _BASE.copy()
    # Right arm extends straight forward; left guard maintained
    extension[14] = [ 0.35, -1.45,  0.20]  # right elbow extends
    extension[16] = [ 0.40, -1.48,  0.55]  # right wrist at full extension
    extension[18] = [ 0.42, -1.49,  0.57]  # right pinky
    extension[20] = [ 0.41, -1.49,  0.58]  # right index
    extension[22] = [ 0.41, -1.48,  0.56]  # right thumb
    # Slight hip rotation: right hip rotates forward
    extension[23] = [-0.10,  0.00, -0.05]
    extension[24] = [ 0.14,  0.00,  0.05]
    # Left guard remains in place
    extension[15] = [-0.20, -1.60,  0.20]

    seq = _build_sequence([
        (0,  guard),
        (9,  guard),
        (29, extension),
        (44, guard),
        (59, guard),
    ])
    return _add_noise(seq, seed=13)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_OUTPUT_DIR,
        help="Directory to write .npy files into (default: backend/data/references/).",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    techniques = {
        "front_kick":      _make_front_kick,
        "roundhouse_kick": _make_roundhouse_kick,
        "straight_punch":  _make_straight_punch,
    }

    for slug, fn in techniques.items():
        seq = fn()  # (N_FRAMES, 33, 3)
        flat = seq.reshape(_N_FRAMES, _N_FEATURES)  # (N_FRAMES, 99)
        out_path = output_dir / f"{slug}.npy"
        np.save(out_path, flat)
        print(f"Saved {out_path}  shape={flat.shape}  dtype={flat.dtype}")

    print("Done.")


if __name__ == "__main__":
    main()
