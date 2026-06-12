"""Build per-technique reference templates from real extracted skeletons.

Phase 2 pipeline step: consumes the smoothed, normalised landmark archives
written by ``data_pipeline.extractor`` (``data/landmarks/<technique>/*.npz``)
and produces the ``(n_frames, 99)`` reference templates consumed by
``backend.scoring.dtw_aligner`` under ``backend/data/references/``.

Per clip:
  1. Quality screen — reject clips with too many undetected frames, low
     landmark visibility, an implausible rep duration, or motion that does
     not look like the technique (e.g. the foot never rises for a kick).
  2. Side canonicalisation — detect the active limb; if the clip executes
     left-sided, mirror the skeleton so every candidate is right-sided
     (matching the established template convention).
  3. Rep segmentation — locate chamber → extension → retraction with
     ``backend.vision.segment.find_rep_window`` and slice that window
     (plus a small pad) out of the clip.

Per technique:
  4. Medoid selection — compute pairwise DTW distances between all accepted
     rep slices and pick the one closest to all others.  A medoid (a real,
     coherent execution) is preferred over a frame-wise average, which smears
     misaligned motions into physically impossible poses.
  5. Persist ``<slug>.npy`` + ``<slug>.meta.json`` with full provenance, and
     render an MP4 stick-figure preview of every accepted rep into
     ``data/review_previews/`` for human review.

Existing templates are only replaced when at least one clip survives
screening; otherwise the previous (synthetic) template is left untouched and
the failure is reported.

Usage::

    python scripts/build_reference_templates.py [--landmarks-dir data/landmarks]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from dtaidistance import dtw_ndim
from scipy.signal import find_peaks

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.vision.segment import SegmentationError, find_rep_window  # noqa: E402

_OUTPUT_DIR = ROOT / "backend" / "data" / "references"
_PREVIEWS_DIR = ROOT / "data" / "review_previews"

_RIGHT_ANKLE, _LEFT_ANKLE = 28, 27
_RIGHT_WRIST, _LEFT_WRIST = 16, 15

# Landmark index pairs swapped when mirroring left <-> right.
_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

_END_EFFECTORS = {
    "front_kick": (_LEFT_ANKLE, _RIGHT_ANKLE),
    "roundhouse_kick": (_LEFT_ANKLE, _RIGHT_ANKLE),
    "straight_punch": (_LEFT_WRIST, _RIGHT_WRIST),
}

# Screening thresholds (normalised units: torso length == 1, y grows downward,
# hips at the origin; a standing ankle sits near y = +1).
_MAX_ZERO_FRAME_FRAC = 0.20
_MIN_VISIBILITY = 0.45
_MIN_REP_SECONDS, _MAX_REP_SECONDS = 0.25, 5.0
_KICK_MIN_FOOT_RISE_Y = 0.40   # ankle must rise to y <= this inside the rep
_PUNCH_MIN_WRIST_TRAVEL = 0.30  # wrist displacement range inside the rep


def _mirror(coords: np.ndarray) -> np.ndarray:
    """Reflect a (T, 33, 3) skeleton left<->right (negate x, swap side pairs)."""
    out = coords.copy()
    out[:, :, 0] *= -1.0
    for a, b in _MIRROR_PAIRS:
        out[:, [a, b], :] = out[:, [b, a], :]
    return out


def _active_side_is_left(coords: np.ndarray, technique: str) -> bool:
    """True when the left end-effector moves more than the right one."""
    left_idx, right_idx = _END_EFFECTORS[technique]
    left_travel = np.abs(np.diff(coords[:, left_idx, :], axis=0)).sum()
    right_travel = np.abs(np.diff(coords[:, right_idx, :], axis=0)).sum()
    return left_travel > right_travel


def _candidate_peak_frames(
    coords: np.ndarray, timestamps: np.ndarray, technique: str, max_peaks: int = 8
) -> list[int]:
    """Frame indices of the strongest end-effector velocity peaks in the clip.

    Long instructional clips mix talking with several demonstrations;
    ``find_rep_window`` assumes a single-rep clip, so we first locate each
    candidate strike by its velocity peak and segment locally around it.
    """
    dt = float(np.median(np.diff(timestamps))) or 1 / 30.0
    left_idx, right_idx = _END_EFFECTORS[technique]
    speeds = []
    for idx in (left_idx, right_idx):
        v = np.linalg.norm(np.diff(coords[:, idx, :], axis=0), axis=1) / dt
        speeds.append(v)
    speed = np.maximum(*speeds)
    if speed.max() <= 0:
        return []

    min_gap = max(1, int(round(1.0 / dt)))  # >= 1 s between distinct reps
    peaks, props = find_peaks(speed, height=0.4 * speed.max(), distance=min_gap)
    order = np.argsort(props["peak_heights"])[::-1]
    return [int(peaks[i]) for i in order[:max_peaks]]


def _extract_reps(
    coords: np.ndarray,
    visibility: np.ndarray,
    timestamps: np.ndarray,
    technique: str,
) -> tuple[list[tuple[np.ndarray, dict]], dict[str, int]]:
    """Harvest every plausible rep from a clip.

    Returns ``(reps, reject_counts)`` where each rep is ``(slice, info)`` and
    *reject_counts* tallies why candidate peaks were discarded.
    """
    n = coords.shape[0]
    dt = float(np.median(np.diff(timestamps))) or 1 / 30.0
    half_window = int(round(2.5 / dt))

    reps: list[tuple[np.ndarray, dict]] = []
    rejects: dict[str, int] = {}
    used: list[tuple[int, int]] = []

    def _reject(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    for peak in _candidate_peak_frames(coords, timestamps, technique):
        w0, w1 = max(0, peak - half_window), min(n, peak + half_window)
        sub = coords[w0:w1]

        zero_frac = float(np.all(sub.reshape(len(sub), -1) == 0.0, axis=1).mean())
        if zero_frac > _MAX_ZERO_FRAME_FRAC:
            _reject("pose undetected around peak")
            continue

        mirrored = _active_side_is_left(sub, technique)
        if mirrored:
            sub = _mirror(sub)

        try:
            chamber, extension, retraction = find_rep_window(sub, technique)
        except SegmentationError:
            _reject("segmentation failed")
            continue

        duration = (retraction - chamber) * dt
        if not _MIN_REP_SECONDS <= duration <= _MAX_REP_SECONDS:
            _reject("implausible rep duration")
            continue

        pad = max(2, (retraction - chamber) // 10)
        start, end = max(0, chamber - pad), min(len(sub), retraction + pad + 1)
        abs_start, abs_end = w0 + start, w0 + end
        if any(min(abs_end, e) - max(abs_start, s) > (abs_end - abs_start) // 2
               for s, e in used):
            _reject("overlaps an already-harvested rep")
            continue

        rep = sub[start:end]

        left_idx, right_idx = _END_EFFECTORS[technique]
        vis = float(visibility[abs_start:abs_end,
                               [left_idx, right_idx, 23, 24]].mean())
        if vis < _MIN_VISIBILITY:
            _reject("low landmark visibility")
            continue

        if technique.endswith("_kick"):
            foot_min_y = float(rep[:, _RIGHT_ANKLE, 1].min())
            if foot_min_y > _KICK_MIN_FOOT_RISE_Y:
                _reject("foot never rises")
                continue
        else:
            wrist = rep[:, _RIGHT_WRIST, :]
            travel = float(np.linalg.norm(wrist.max(axis=0) - wrist.min(axis=0)))
            if travel < _PUNCH_MIN_WRIST_TRAVEL:
                _reject("wrist barely moves")
                continue

        used.append((abs_start, abs_end))
        reps.append((rep, {
            "frames": int(rep.shape[0]),
            "window": [int(abs_start), int(w0 + extension), int(abs_end - 1)],
            "rep_seconds": round(float(duration), 2),
            "mirrored": bool(mirrored),
            "mean_visibility": round(vis, 3),
        }))

    return reps, rejects


def _medoid_index(reps: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """Index of the rep with the lowest mean DTW distance to all others."""
    k = len(reps)
    flat = [r.reshape(r.shape[0], -1).astype(np.double) for r in reps]
    dist = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = dtw_ndim.distance(flat[i], flat[j])
            dist[i, j] = dist[j, i] = d
    mean_dist = dist.sum(axis=1) / max(k - 1, 1)
    return int(mean_dist.argmin()), mean_dist


def _render_previews(reps: dict[str, np.ndarray], technique: str) -> list[str]:
    """Render an MP4 stick-figure preview per accepted rep; return paths."""
    from data_pipeline.review import _render_mp4  # lazy: needs matplotlib

    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for clip_id, rep in reps.items():
        try:
            out = _render_mp4(rep, clip_id, technique)
            paths.append(str(out))
        except Exception as exc:  # noqa: BLE001
            print(f"    [preview failed] {clip_id}: {exc}")
    return paths


def build_technique(technique: str, landmarks_dir: pathlib.Path) -> bool:
    tech_dir = landmarks_dir / technique
    npz_files = sorted(tech_dir.glob("*.npz")) if tech_dir.exists() else []
    print(f"\n{technique}: {len(npz_files)} candidate skeleton(s)")
    if not npz_files:
        print("  No landmark files — keeping the existing template.")
        return False

    accepted: dict[str, np.ndarray] = {}
    accepted_info: dict[str, dict] = {}
    rejected: dict[str, dict[str, int]] = {}

    for npz_path in npz_files:
        clip_id = npz_path.stem
        data = np.load(npz_path, allow_pickle=False)
        reps, rejects = _extract_reps(
            data["coords"], data["visibility"], data["timestamps"], technique
        )
        if rejects:
            rejected[clip_id] = rejects
        for k, (rep, info) in enumerate(reps):
            rep_id = f"{clip_id}#{k}"
            accepted[rep_id] = rep
            accepted_info[rep_id] = info
        summary = ", ".join(f"{r}×{c}" for r, c in rejects.items()) or "none"
        print(f"  {clip_id}: {len(reps)} rep(s) harvested (rejected: {summary})")

    if not accepted:
        print("  All candidates rejected — keeping the existing template.")
        return False

    # Keep pairwise DTW tractable: cap at the highest-visibility reps.
    _MAX_REPS = 24
    if len(accepted) > _MAX_REPS:
        keep = sorted(accepted, key=lambda r: accepted_info[r]["mean_visibility"],
                      reverse=True)[:_MAX_REPS]
        dropped = len(accepted) - _MAX_REPS
        accepted = {r: accepted[r] for r in keep}
        accepted_info = {r: accepted_info[r] for r in keep}
        print(f"  Capped to the {_MAX_REPS} highest-visibility reps "
              f"({dropped} dropped).")

    clip_ids = list(accepted)
    medoid_idx, mean_dists = _medoid_index([accepted[c] for c in clip_ids])
    medoid_id = clip_ids[medoid_idx]
    template = accepted[medoid_id].reshape(accepted[medoid_id].shape[0], -1)
    template = template.astype(np.float64)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"{technique}.npy"
    np.save(out_path, template)

    preview_paths = _render_previews(accepted, technique)

    meta = {
        "source": "scraped-youtube",
        "built_by": "scripts/build_reference_templates.py",
        "technique": technique,
        "frames": int(template.shape[0]),
        "candidates": len(npz_files),
        "accepted_reps": len(accepted),
        "medoid_rep": medoid_id,
        "reps": {
            cid: {**accepted_info[cid],
                  "mean_dtw_distance": round(float(mean_dists[i]), 3)}
            for i, cid in enumerate(clip_ids)
        },
        "rejected": rejected,
        "previews": preview_paths,
        "human_review": "pending",
        "note": (
            "Template is the medoid rep of auto-screened scraped clips. "
            "Spot-check the previews in data/review_previews/ before "
            "treating scores as gold-standard."
        ),
    }
    (out_path.with_suffix(".meta.json")).write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(f"  Template ← medoid '{medoid_id}' "
          f"({template.shape[0]} frames, {len(accepted)} accepted clips)")
    print(f"  Saved {out_path} and meta sidecar.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks-dir", type=pathlib.Path,
                        default=ROOT / "data" / "landmarks")
    args = parser.parse_args()

    techniques = ["front_kick", "roundhouse_kick", "straight_punch"]
    built = [t for t in techniques if build_technique(t, args.landmarks_dir)]
    print(f"\nBuilt {len(built)}/{len(techniques)} templates: {built}")
    if len(built) < len(techniques):
        sys.exit(1)


if __name__ == "__main__":
    main()
