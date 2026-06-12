"""Skeleton extraction, smoothing, normalisation, and storage.

Processing pipeline per accepted staged clip:
  1. Open video with OpenCV and run MediaPipe PoseLandmarker frame-by-frame.
  2. Collect world-space 3D landmark coordinates (N×33×3) and visibility
     scores (N×33), plus per-frame timestamps.
  3. Apply a Savitzky-Golay smoothing pass along the time axis.
  4. Normalise: translate so the hip midpoint is the origin each frame, then
     scale so the median torso height (hip-midpoint → shoulder-midpoint) is 1.0.
  5. Persist as a compressed NumPy .npz file under
     data/landmarks/<technique>/<clip_id>.npz.

Staged-clip registry
--------------------
A JSON manifest at ``data/staged/manifest.json`` lists candidate clips:

    [
      {
        "technique": "front_kick",
        "clip_id": "clip_001",
        "path": "data/staged/front_kick/clip_001.mp4",
        "accepted": true
      },
      ...
    ]

Only entries with ``"accepted": true`` are processed by the ``extract``
command.  Raw video files are never written to ``data/landmarks/``.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_lite.task")
DEFAULT_STAGED_MANIFEST = Path("data/staged/manifest.json")
DEFAULT_LANDMARKS_DIR = Path("data/landmarks")

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


# ── Model helpers ─────────────────────────────────────────────────────────────

def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the MediaPipe pose landmarker model if it is not present.

    Parameters
    ----------
    model_path:
        Destination path for the ``.task`` model bundle.

    Returns
    -------
    Path
        The (now-existing) model path.
    """
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose landmarker model → {model_path} …")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete.")
    return model_path


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_landmarks(
    video_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 3D pose landmarks from every frame of a video.

    Uses MediaPipe PoseLandmarker in VIDEO running mode so that landmark
    tracking stays consistent across frames.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    model_path:
        Path to the MediaPipe pose landmarker ``.task`` bundle.  The model is
        downloaded automatically on first use if not already present.

    Returns
    -------
    coords : ndarray, shape (N, 33, 3), float32
        World-space (x, y, z) landmark coordinates per frame, in metres.
    visibility : ndarray, shape (N, 33), float32
        Per-landmark visibility confidence score ∈ [0, 1] per frame.
    timestamps : ndarray, shape (N,), float64
        Frame timestamps in seconds (derived from the video frame rate).

    Raises
    ------
    IOError
        If the video file cannot be opened.
    ValueError
        If no frames could be read from the video.
    """
    ensure_model(model_path)

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_coords: list[np.ndarray] = []
    all_visibility: list[np.ndarray] = []
    all_timestamps: list[float] = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1_000.0 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                lms = result.pose_world_landmarks[0]  # single-person clip
                frame_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
                )
                frame_vis = np.array(
                    [lm.visibility if lm.visibility is not None else 0.0 for lm in lms],
                    dtype=np.float32,
                )
            else:
                # No pose detected — fill with zeros so shapes stay consistent.
                frame_coords = np.zeros((33, 3), dtype=np.float32)
                frame_vis = np.zeros(33, dtype=np.float32)

            all_coords.append(frame_coords)
            all_visibility.append(frame_vis)
            all_timestamps.append(timestamp_ms / 1_000.0)
            frame_idx += 1

    cap.release()

    if not all_coords:
        raise ValueError(f"No frames could be read from {video_path}")

    return (
        np.stack(all_coords),       # (N, 33, 3)
        np.stack(all_visibility),   # (N, 33)
        np.array(all_timestamps, dtype=np.float64),  # (N,)
    )


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_landmarks(
    coords: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter along the time axis of landmark coords.

    Parameters
    ----------
    coords:
        Raw landmark coordinates, shape (N, 33, 3).
    window_length:
        Length of the filter window (must be odd and ≥ ``polyorder + 1``).
        Automatically clamped to the sequence length when N is small.
    polyorder:
        Polynomial order for the filter.

    Returns
    -------
    ndarray, shape (N, 33, 3), float32
        Smoothed landmark coordinates.
    """
    n_frames = coords.shape[0]

    # Clamp window_length to the sequence length; keep it odd.
    wl = min(window_length, n_frames)
    if wl % 2 == 0:
        wl -= 1
    # Must be strictly greater than polyorder.
    min_wl = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    wl = max(wl, min_wl)

    if wl > n_frames:
        # Sequence too short to smooth — return a copy unchanged.
        return coords.astype(np.float32).copy()

    smoothed = savgol_filter(
        coords, window_length=wl, polyorder=polyorder, axis=0
    )
    return smoothed.astype(np.float32)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """Normalise skeleton coordinates to a canonical pose space.

    Two transformations are applied:

    1. **Translation** — subtract the hip midpoint from every landmark so that
       the midpoint of the left and right hips sits at the origin for every
       frame independently.
    2. **Scale** — divide by the *median* torso height across all frames, where
       torso height is the Euclidean distance from the (now-zero) hip midpoint
       to the shoulder midpoint.  Using the median (rather than per-frame
       torso height) avoids magnifying noise on frames where the shoulders are
       partially occluded.

    After this transform:
    - The mean hip-midpoint position across all frames is (0, 0, 0).
    - The median torso height equals 1.0.

    Parameters
    ----------
    coords:
        Landmark coordinates, shape (N, 33, 3).

    Returns
    -------
    ndarray, shape (N, 33, 3), float32
        Normalised landmark coordinates.

    Raises
    ------
    ValueError
        If the median torso height is effectively zero (degenerate input).
    """
    # 1. Hip-centred translation (per frame).
    hip_mid = (coords[:, _LEFT_HIP, :] + coords[:, _RIGHT_HIP, :]) / 2.0  # (N, 3)
    centred = coords - hip_mid[:, np.newaxis, :]  # broadcast over 33 landmarks

    # 2. Torso-height scale (global median for stability).
    shoulder_mid = (
        centred[:, _LEFT_SHOULDER, :] + centred[:, _RIGHT_SHOULDER, :]
    ) / 2.0  # (N, 3)
    torso_heights = np.linalg.norm(shoulder_mid, axis=1)  # (N,)
    scale = float(np.median(torso_heights))

    if scale < 1e-6:
        raise ValueError(
            "Median torso height is near zero — the skeleton input appears "
            "degenerate (all landmarks collapsed to a point)."
        )

    normalised = centred / scale
    return normalised.astype(np.float32)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_accepted_clips(
    manifest_path: Path = DEFAULT_STAGED_MANIFEST,
) -> list[dict]:
    """Return the list of accepted clip entries from the staged manifest.

    The manifest is a JSON array of objects with at minimum the fields:
    ``technique`` (str), ``clip_id`` (str), ``path`` (str), ``accepted`` (bool).

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json``.

    Returns
    -------
    list[dict]
        Entries where ``"accepted"`` is ``True``.

    Raises
    ------
    FileNotFoundError
        If the manifest does not exist.
    """
    with manifest_path.open() as fh:
        clips = json.load(fh)
    return [c for c in clips if c.get("accepted", False)]


# ── Top-level extraction pipeline ─────────────────────────────────────────────

def process_clip(
    clip: dict,
    landmarks_dir: Path = DEFAULT_LANDMARKS_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    overwrite: bool = False,
) -> Path:
    """Run the full extraction pipeline for a single accepted clip.

    Steps: extract → smooth → normalise → save .npz.

    The output file is stored at::

        <landmarks_dir>/<technique>/<clip_id>.npz

    and contains three arrays:

    - ``coords``      shape (N, 33, 3), float32 — normalised 3-D coordinates.
    - ``visibility``  shape (N, 33), float32    — per-landmark visibility scores.
    - ``timestamps``  shape (N,), float64       — frame timestamps in seconds.

    Parameters
    ----------
    clip:
        Manifest entry dict with keys ``technique``, ``clip_id``, ``path``.
    landmarks_dir:
        Root output directory (``data/landmarks`` by default).
    model_path:
        Path to the MediaPipe pose landmarker model.
    overwrite:
        If ``False`` (default) and the output .npz already exists, skip
        re-processing and return the existing path.

    Returns
    -------
    Path
        Path of the written (or pre-existing) .npz file.
    """
    technique: str = clip["technique"]
    clip_id: str = clip["clip_id"]
    video_path = Path(clip["path"])

    out_dir = landmarks_dir / technique
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} already exists.")
        return out_path

    print(f"  Processing {video_path} …")
    coords, visibility, timestamps = extract_landmarks(video_path, model_path)
    coords = smooth_landmarks(coords)
    coords = normalize_landmarks(coords)

    np.savez_compressed(
        out_path,
        coords=coords,
        visibility=visibility,
        timestamps=timestamps,
    )
    print(f"  Saved → {out_path}  (frames={len(timestamps)})")
    return out_path


def extract_all(
    manifest_path: Path = DEFAULT_STAGED_MANIFEST,
    landmarks_dir: Path = DEFAULT_LANDMARKS_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    overwrite: bool = False,
) -> list[Path]:
    """Process every accepted clip listed in the staged manifest.

    Parameters
    ----------
    manifest_path:
        Path to the JSON manifest describing staged clips.
    landmarks_dir:
        Root directory under which .npz files are written.
    model_path:
        Path to the MediaPipe pose landmarker model bundle.
    overwrite:
        Re-process clips that already have an output file when ``True``.

    Returns
    -------
    list[Path]
        Paths of all .npz files produced (or already present).
    """
    accepted = load_accepted_clips(manifest_path)
    if not accepted:
        print("No accepted clips found in manifest.")
        return []

    print(f"Found {len(accepted)} accepted clip(s).")
    results: list[Path] = []
    for clip in accepted:
        try:
            out = process_clip(clip, landmarks_dir, model_path, overwrite)
            results.append(out)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {clip.get('clip_id', '?')}: {exc}")

    return results
