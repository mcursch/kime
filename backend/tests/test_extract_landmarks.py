"""Unit tests for scripts/extract_landmarks.py.

The tests exercise the full pipeline against a synthetically-generated
5-second sample video clip so the test suite is self-contained (no large
binary assets required).  For each frame the expected output shape is
validated:

  - One JSON entry per video frame
  - Each entry is a list
  - Non-empty entries contain exactly 33 landmark dicts
  - Every landmark dict has the keys x, y, z, and visibility

A separate set of tests (``test_detection_code_path_*``) use a mocked
landmarker to exercise the non-empty detection branch of ``extract()``
without requiring a real person in the video.  An optional integration
test (``test_real_person_video_detection_rate``) can be activated by
placing a video file containing a fully-visible person at
``data/test_person.mp4``; it is skipped automatically when the file is
absent.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Make the scripts package importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


# ---------------------------------------------------------------------------
# Detection-code-path tests (mocked landmarker)
#
# These tests exercise the ``if detection.pose_landmarks:`` branch of
# ``extract()`` without requiring a real person in the video.  They use a
# mocked landmarker that unconditionally returns 33 fake landmarks per frame,
# confirming that the serialisation logic (float conversion, key names, count
# assertion) is correct.
# ---------------------------------------------------------------------------


def _make_fake_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0, visibility: float = 0.99):
    """Return a MagicMock that mimics a MediaPipe NormalizedLandmark."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


def _make_fake_landmarker(num_frames: int = 5):
    """Return a context-manager-compatible mock landmarker.

    Every call to ``detect()`` returns a result whose ``pose_landmarks``
    contains one person with 33 fake landmarks, simulating a fully-visible
    person in every frame.
    """
    fake_landmarks_per_person = [_make_fake_landmark(x=i * 0.01) for i in range(_NUM_LANDMARKS)]

    fake_detection = MagicMock()
    fake_detection.pose_landmarks = [fake_landmarks_per_person]

    landmarker = MagicMock()
    landmarker.detect.return_value = fake_detection
    # Support use as a context manager (``with _build_landmarker(...) as l``)
    landmarker.__enter__ = MagicMock(return_value=landmarker)
    landmarker.__exit__ = MagicMock(return_value=False)
    return landmarker


@pytest.fixture(scope="module")
def mocked_detection_output(tmp_path_factory: pytest.TempPathFactory):
    """Run extract() with a mocked landmarker; return (data, frame_count)."""
    tmp_dir = tmp_path_factory.mktemp("mocked_data")
    video_path = tmp_dir / "person_clip.mp4"
    fps, duration_s = 10, 1
    frame_count = _create_sample_video(video_path, fps=fps, duration_s=duration_s)

    # Fake model file — extract() checks existence before calling _build_landmarker
    fake_model = tmp_dir / "fake_model.task"
    fake_model.write_bytes(b"")

    import scripts.extract_landmarks as el

    with patch.object(el, "_build_landmarker", return_value=_make_fake_landmarker(frame_count)):
        out_path = el.extract(video_path, fake_model, tmp_dir)

    with out_path.open() as fh:
        data = json.load(fh)

    return data, frame_count


def test_detection_code_path_all_frames_non_empty(mocked_detection_output) -> None:
    """When every frame detects a person, no entry should be an empty list.

    This directly validates the acceptance criterion:
    'No frame produces an empty landmark list for a clip where a person is
    fully visible.'
    """
    data, frame_count = mocked_detection_output
    assert len(data) == frame_count
    empty_frames = [i for i, entry in enumerate(data) if not entry]
    assert empty_frames == [], (
        f"Expected all {frame_count} frames to have landmarks but frames "
        f"{empty_frames} were empty."
    )


def test_detection_code_path_landmark_count(mocked_detection_output) -> None:
    """Each detected frame must contain exactly 33 landmark dicts."""
    data, _ = mocked_detection_output
    for frame_idx, entry in enumerate(data):
        assert len(entry) == _NUM_LANDMARKS, (
            f"Frame {frame_idx}: expected {_NUM_LANDMARKS} landmarks, got {len(entry)}"
        )


def test_detection_code_path_landmark_keys(mocked_detection_output) -> None:
    """Every landmark dict must contain x, y, z, and visibility as floats."""
    data, _ = mocked_detection_output
    for frame_idx, entry in enumerate(data):
        for lm_idx, lm in enumerate(entry):
            missing = _LANDMARK_KEYS - set(lm.keys())
            assert not missing, (
                f"Frame {frame_idx}, landmark {lm_idx}: missing keys {missing}"
            )
            for key in _LANDMARK_KEYS:
                assert isinstance(lm[key], float), (
                    f"Frame {frame_idx}, landmark {lm_idx}, key '{key}': "
                    f"expected float, got {type(lm[key]).__name__}"
                )


# ---------------------------------------------------------------------------
# Optional real-person integration test
#
# Place a video with a fully-visible person at data/test_person.mp4 and
# ensure the model file is present to activate this test.  It is skipped
# automatically in environments where the file is absent.
# ---------------------------------------------------------------------------

_REAL_PERSON_VIDEO = _REPO_ROOT / "data" / "test_person.mp4"
_REAL_PERSON_DETECTION_THRESHOLD = 0.95  # fraction of frames that must have landmarks


@pytest.mark.skipif(
    not (_REAL_PERSON_VIDEO.exists() and _MODEL.exists()),
    reason=(
        "Skipped: place a video with a fully-visible person at data/test_person.mp4 "
        "and ensure the model is present to run this test."
    ),
)
def test_real_person_video_detection_rate(tmp_path: Path) -> None:
    """For a clip with a fully-visible person, >95 % of frames must be non-empty.

    This is the end-to-end acceptance criterion exercised against real
    MediaPipe inference rather than a mock.
    """
    import scripts.extract_landmarks as el

    out_path = el.extract(_REAL_PERSON_VIDEO, _MODEL, tmp_path)
    with out_path.open() as fh:
        data = json.load(fh)

    non_empty = sum(1 for entry in data if entry)
    fraction = non_empty / len(data) if data else 0.0
    assert fraction >= _REAL_PERSON_DETECTION_THRESHOLD, (
        f"Only {fraction:.1%} of frames had detections; expected "
        f">= {_REAL_PERSON_DETECTION_THRESHOLD:.0%}."
    )
