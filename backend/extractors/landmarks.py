"""Pose landmark extraction using MediaPipe Pose Landmarker.

Each frame yields a list of 33 landmark dicts with keys:
  x, y, z           – normalised image coords + depth
  visibility         – landmark visibility score
  presence           – landmark presence score (MediaPipe ≥0.10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

# Number of pose landmarks MediaPipe returns per frame
_NUM_LANDMARKS = 33


def extract_landmarks(video_path: str | Path) -> list[dict[str, Any]]:
    """Extract pose landmarks from every frame of *video_path*.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the video file.

    Returns
    -------
    list[dict]
        One entry per frame.  Each entry has the shape::

            {
                "frame": <int>,
                "landmarks": [
                    {"x": float, "y": float, "z": float,
                     "visibility": float, "presence": float},
                    ...  # 33 items
                ]
            }

        Frames where no pose is detected are omitted.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    results: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = pose.process(rgb)

            if detection.pose_landmarks:
                landmarks = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                        "presence": getattr(lm, "presence", 1.0),
                    }
                    for lm in detection.pose_landmarks.landmark
                ]
                results.append({"frame": frame_idx, "landmarks": landmarks})

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    logger.info(
        "Extracted landmarks from %d/%d frames in %s",
        len(results),
        frame_idx,
        video_path.name,
    )
    return results
