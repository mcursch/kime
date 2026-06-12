"""Extract MediaPipe Pose Landmarker landmarks from every frame of a video.

Usage
-----
    python scripts/extract_landmarks.py <video_path> [--model <model_path>]

Output
------
    data/<video_stem>_landmarks.json

    A JSON array with one element per frame.  Each element is either:
      - A list of 33 landmark dicts, each with keys x, y, z, visibility
      - An empty list [] when no person was detected in that frame
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Default model path relative to the repository root (parent of scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "models" / "pose_landmarker_lite.task"

NUM_LANDMARKS = 33


def _build_landmarker(model_path: Path) -> vision.PoseLandmarker:
    """Construct a PoseLandmarker configured for image (per-frame) mode."""
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract(video_path: Path, model_path: Path, output_dir: Path) -> Path:
    """Run landmark extraction and write results to *output_dir*.

    Returns
    -------
    Path
        The path of the written JSON file.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_landmarks.json"

    results: list[list[dict]] = []

    try:
        with _build_landmarker(model_path) as landmarker:
            frame_idx = 0
            while True:
                ok, bgr_frame = cap.read()
                if not ok:
                    break

                # MediaPipe expects RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = landmarker.detect(mp_image)

                if detection.pose_landmarks:
                    landmarks = [
                        {
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility),
                        }
                        for lm in detection.pose_landmarks[0]
                    ]
                    assert len(landmarks) == NUM_LANDMARKS, (
                        f"Frame {frame_idx}: expected {NUM_LANDMARKS} landmarks, "
                        f"got {len(landmarks)}"
                    )
                else:
                    landmarks = []

                results.append(landmarks)
                frame_idx += 1
    finally:
        cap.release()

    with output_path.open("w") as fh:
        json.dump(results, fh)

    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose Landmarker landmarks from a video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL,
        help="Path to the pose_landmarker_lite.task model file "
        f"(default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Directory to write the landmark JSON file (default: <repo_root>/data).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = extract(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
    )
    print(f"Wrote landmarks to {output_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
