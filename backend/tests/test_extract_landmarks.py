"""Unit tests for scripts/extract_landmarks.py.

The tests exercise the full pipeline against a synthetically-generated
5-second sample video clip so the test suite is self-contained (no large
binary assets required).  For each frame the expected output shape is
validated:

  - One JSON entry per video frame
  - Each entry is a list
  - Non-empty entries contain exactly 33 landmark dicts
  - Every landmark dict has the keys x, y, z, and visibility
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "extract_landmarks.py"
_MODEL = _REPO_ROOT / "models" / "pose_landmarker_lite.task"

# Required landmark keys
_LANDMARK_KEYS = {"x", "y", "z", "visibility"}
_NUM_LANDMARKS = 33


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_sample_video(path: Path, fps: int = 30, duration_s: int = 5) -> int:
    """Write a small synthetic video to *path* and return the frame count.

    The video consists of solid-colour frames that shift hue each second so
    that codec motion estimation has something to work with.  No person is
    embedded; this clip is intended only to validate pipeline mechanics and
    output structure.
    """
    width, height = 320, 240
    total_frames = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path}")

    for i in range(total_frames):
        # Cycle through a simple colour gradient per frame
        hue = int((i / total_frames) * 180)
        hsv = np.full((height, width, 3), (hue, 200, 200), dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(bgr)

    writer.release()
    return total_frames


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pytest fixture that creates a temporary synthetic video clip."""
    tmp_dir = tmp_path_factory.mktemp("video")
    video_path = tmp_dir / "sample_clip.mp4"
    _create_sample_video(video_path)
    return video_path


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pytest fixture providing a temporary output directory."""
    return tmp_path_factory.mktemp("data")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL.exists(),
    reason=f"Model file not found: {_MODEL}. Download pose_landmarker_lite.task first.",
)
def test_script_exits_zero(sample_video: Path, output_dir: Path) -> None:
    """The script must exit with code 0 for a valid video."""
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(sample_video),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.skipif(
    not _MODEL.exists(),
    reason=f"Model file not found: {_MODEL}. Download pose_landmarker_lite.task first.",
)
def test_output_file_created_under_data_dir(sample_video: Path, output_dir: Path) -> None:
    """A JSON file must appear in the designated output directory."""
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(sample_video),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=True,
    )
    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) >= 1, f"No JSON files found in {output_dir}"


@pytest.mark.skipif(
    not _MODEL.exists(),
    reason=f"Model file not found: {_MODEL}. Download pose_landmarker_lite.task first.",
)
class TestOutputShape:
    """Validates the shape of the landmark JSON produced by the script."""

    @pytest.fixture(scope="class", autouse=True)
    def run_script(
        self, sample_video: Path, output_dir: Path
    ) -> None:  # type: ignore[override]
        """Run the extraction script once; all tests in this class share the result."""
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                str(sample_video),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            check=True,
        )
        # Cache the expected frame count on the class
        cap = cv2.VideoCapture(str(sample_video))
        self.__class__._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        json_path = output_dir / f"{sample_video.stem}_landmarks.json"
        with json_path.open() as fh:
            self.__class__._data = json.load(fh)

    def test_one_entry_per_frame(self) -> None:
        """JSON array length must equal the number of frames in the video."""
        assert len(self._data) == self._frame_count, (
            f"Expected {self._frame_count} entries, got {len(self._data)}"
        )

    def test_each_entry_is_a_list(self) -> None:
        """Every frame entry must be a list (possibly empty when no person detected)."""
        for idx, entry in enumerate(self._data):
            assert isinstance(entry, list), (
                f"Frame {idx}: expected list, got {type(entry).__name__}"
            )

    def test_detected_frames_have_33_landmarks(self) -> None:
        """Non-empty entries must have exactly 33 landmark objects."""
        for idx, entry in enumerate(self._data):
            if entry:
                assert len(entry) == _NUM_LANDMARKS, (
                    f"Frame {idx}: expected {_NUM_LANDMARKS} landmarks, got {len(entry)}"
                )

    def test_landmark_objects_have_required_keys(self) -> None:
        """Every landmark dict must contain x, y, z, and visibility."""
        for frame_idx, entry in enumerate(self._data):
            for lm_idx, lm in enumerate(entry):
                missing = _LANDMARK_KEYS - set(lm.keys())
                assert not missing, (
                    f"Frame {frame_idx}, landmark {lm_idx}: missing keys {missing}"
                )

    def test_landmark_values_are_numeric(self) -> None:
        """x, y, z, and visibility must be numeric (float or int)."""
        for frame_idx, entry in enumerate(self._data):
            for lm_idx, lm in enumerate(entry):
                for key in _LANDMARK_KEYS:
                    assert isinstance(lm[key], (int, float)), (
                        f"Frame {frame_idx}, landmark {lm_idx}, key '{key}': "
                        f"expected numeric, got {type(lm[key]).__name__}"
                    )
