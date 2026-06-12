"""
Single-person quality filter using MediaPipe Pose Landmarker (Tasks API).

Algorithm (per clip):
1. Sample ``n_samples`` frames evenly across the clip duration.
2. For each sampled frame, run MediaPipe PoseLandmarker with ``num_poses=4``
   so that up to four skeletons can be detected simultaneously.
3. Majority vote across sampled frames:
   - If the fraction of frames where *no* pose was detected ≥ ``rejection_threshold``
     → reject as "no pose detected".
   - If the fraction of frames where *multiple* poses were detected ≥
     ``rejection_threshold`` → reject as "multi-person detected".
   - Otherwise → accept.

Rejected clips are moved to ``<staging_dir>/rejected/<label>/`` and a warning
is logged.  Accepted clips stay in ``<staging_dir>/<label>/``.

Model
-----
The PoseLandmarker requires a ``.task`` model bundle.  Pass an explicit
``model_path`` to :func:`filter_clip` / :func:`filter_all`, or set the
``KIME_POSE_MODEL`` environment variable.  If neither is provided the model is
downloaded automatically to ``~/.cache/kime/pose_landmarker_lite.task``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import urllib.request
from typing import Generator

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MODEL_CACHE = pathlib.Path.home() / ".cache" / "kime" / "pose_landmarker_lite.task"


def _resolve_model(model_path: str | pathlib.Path | None) -> pathlib.Path:
    """Return the path to the pose landmarker model, downloading if needed."""
    if model_path is not None:
        return pathlib.Path(model_path)

    env_path = os.environ.get("KIME_POSE_MODEL")
    if env_path:
        return pathlib.Path(env_path)

    if not _MODEL_CACHE.exists():
        logger.info("Downloading PoseLandmarker model → %s", _MODEL_CACHE)
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)

    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_frames(
    video_path: pathlib.Path, n_samples: int
) -> Generator[tuple[int, "cv2.Mat"], None, None]:
    """Yield ``(frame_index, BGR frame)`` for *n_samples* evenly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    step = max(1, total_frames // n_samples)
    indices = list(range(0, total_frames, step))[:n_samples]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            yield idx, frame

    cap.release()


def _count_poses(
    landmarker: mp_vision.PoseLandmarker,
    frame_bgr: "cv2.Mat",
) -> int:
    """Return the number of pose skeletons detected in *frame_bgr*."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    return len(result.pose_landmarks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REJECTION_DIR_NAME = "rejected"

# Maximum number of people the landmarker will attempt to find per frame.
# Using 4 means clips with a crowd are caught even if not all people are
# fully visible.
_NUM_POSES = 4


def filter_clip(
    clip_path: pathlib.Path,
    staging_dir: pathlib.Path,
    n_samples: int = 10,
    rejection_threshold: float = 0.5,
    model_path: str | pathlib.Path | None = None,
) -> bool:
    """Evaluate *clip_path* and move it to ``rejected/`` if it fails quality.

    Parameters
    ----------
    clip_path:
        Path to the video file to evaluate.
    staging_dir:
        Root staging directory.  The ``rejected/`` sub-directory is placed here.
    n_samples:
        Number of frames to sample from the clip for the majority vote.
    rejection_threshold:
        Fraction of sampled frames that must trigger a condition for the clip
        to be rejected (default 0.5 → simple majority).
    model_path:
        Optional explicit path to the ``.task`` model bundle.  Falls back to
        ``KIME_POSE_MODEL`` env-var then an auto-downloaded cache.

    Returns
    -------
    bool
        ``True`` if the clip is accepted, ``False`` if it was rejected.
    """
    frames = list(_sample_frames(clip_path, n_samples))

    if not frames:
        _reject(clip_path, staging_dir, "no frames could be sampled")
        return False

    resolved_model = _resolve_model(model_path)
    base_opts = mp_python.BaseOptions(model_asset_path=str(resolved_model))
    landmarker_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=_NUM_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    no_pose_count = 0
    multi_person_count = 0

    with mp_vision.PoseLandmarker.create_from_options(landmarker_opts) as landmarker:
        for _idx, frame in frames:
            n_poses = _count_poses(landmarker, frame)
            if n_poses == 0:
                no_pose_count += 1
            elif n_poses > 1:
                multi_person_count += 1

    n = len(frames)
    no_pose_frac = no_pose_count / n
    multi_person_frac = multi_person_count / n

    logger.debug(
        "%s — sampled %d frames: no-pose=%.0f%%, multi-person=%.0f%%",
        clip_path.name,
        n,
        no_pose_frac * 100,
        multi_person_frac * 100,
    )

    # Multi-person takes priority so the logged reason is unambiguous.
    if multi_person_frac >= rejection_threshold:
        _reject(clip_path, staging_dir, "multi-person detected")
        return False

    if no_pose_frac >= rejection_threshold:
        _reject(clip_path, staging_dir, "no pose detected")
        return False

    logger.info("Accepted: %s", clip_path)
    return True


def _reject(
    clip_path: pathlib.Path,
    staging_dir: pathlib.Path,
    reason: str,
) -> None:
    """Move *clip_path* to the ``rejected/`` sub-directory and log the reason."""
    try:
        rel = clip_path.relative_to(staging_dir)
    except ValueError:
        rel = pathlib.Path(clip_path.name)

    dest_dir = staging_dir / REJECTION_DIR_NAME / rel.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / clip_path.name

    shutil.move(str(clip_path), dest)
    logger.warning("Rejected (%s): %s → %s", reason, clip_path, dest)


def filter_all(
    staging_dir: str | pathlib.Path,
    n_samples: int = 10,
    rejection_threshold: float = 0.5,
    model_path: str | pathlib.Path | None = None,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Run the single-person filter on every clip in *staging_dir*.

    Clips already inside the ``rejected/`` sub-directory are skipped so
    repeated runs are safe.

    Parameters
    ----------
    staging_dir:
        Root staging directory containing label sub-directories of video files.
    n_samples:
        Frames to sample per clip.
    rejection_threshold:
        Majority-vote threshold for rejection.
    model_path:
        Optional explicit path to the ``.task`` model bundle.

    Returns
    -------
    tuple[list[pathlib.Path], list[pathlib.Path]]
        ``(accepted, rejected)`` path lists.
    """
    root = pathlib.Path(staging_dir)
    rejected_root = root / REJECTION_DIR_NAME

    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    clips = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in video_extensions
        and rejected_root not in p.parents  # skip already-rejected files
    ]

    accepted: list[pathlib.Path] = []
    rejected: list[pathlib.Path] = []

    for clip in clips:
        ok = filter_clip(
            clip,
            root,
            n_samples=n_samples,
            rejection_threshold=rejection_threshold,
            model_path=model_path,
        )
        if ok:
            accepted.append(clip)
        else:
            rejected.append(clip)

    logger.info(
        "Filter complete: %d accepted, %d rejected (of %d total).",
        len(accepted),
        len(rejected),
        len(clips),
    )
    return accepted, rejected
