"""Unit tests for backend.scoring.dtw_aligner."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from backend.scoring.dtw_aligner import (
    AlignmentResult,
    align_sequence,
    align_to_reference,
    load_reference_template,
    _REFERENCES_DIR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 99  # 33 landmarks × 3 coordinates


def _make_sequence(n_frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_frames, N_FEATURES))


# ---------------------------------------------------------------------------
# align_sequence tests
# ---------------------------------------------------------------------------


class TestAlignSequence:
    def test_returns_alignment_result(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert isinstance(result, AlignmentResult)

    def test_warping_path_is_list_of_tuples(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert isinstance(result.path, list)
        assert len(result.path) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result.path)

    def test_path_indices_in_bounds(self):
        n_query, n_ref = 50, 40
        seq = _make_sequence(n_query)
        ref = _make_sequence(n_ref, seed=1)
        result = align_sequence(seq, ref)
        for i, j in result.path:
            assert 0 <= i < n_query, f"query index {i} out of range"
            assert 0 <= j < n_ref, f"reference index {j} out of range"

    def test_frame_distances_shape_matches_path(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert result.frame_distances.shape == (len(result.path),)

    def test_frame_distances_are_non_negative(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert np.all(result.frame_distances >= 0)

    def test_mean_distance_is_scalar_float(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert isinstance(result.mean_distance, float)

    def test_mean_distance_equals_mean_of_frame_distances(self):
        seq = _make_sequence(50)
        ref = _make_sequence(40, seed=1)
        result = align_sequence(seq, ref)
        assert result.mean_distance == pytest.approx(result.frame_distances.mean())

    def test_identical_sequence_gives_near_zero_distance(self):
        seq = _make_sequence(50)
        result = align_sequence(seq, seq)
        assert result.mean_distance == pytest.approx(0.0, abs=1e-10)

    def test_same_length_sequences(self):
        seq = _make_sequence(30)
        ref = _make_sequence(30, seed=99)
        result = align_sequence(seq, ref)
        assert result.mean_distance > 0

    def test_path_starts_and_ends_at_boundaries(self):
        """DTW warping path must start at (0, 0) and end at (n-1, m-1)."""
        n_query, n_ref = 50, 35
        seq = _make_sequence(n_query)
        ref = _make_sequence(n_ref, seed=7)
        result = align_sequence(seq, ref)
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (n_query - 1, n_ref - 1)

    def test_raises_on_1d_sequence(self):
        seq = np.random.rand(50)
        ref = _make_sequence(40)
        with pytest.raises(ValueError, match="2-D"):
            align_sequence(seq, ref)

    def test_raises_on_1d_reference(self):
        seq = _make_sequence(50)
        ref = np.random.rand(40)
        with pytest.raises(ValueError, match="2-D"):
            align_sequence(seq, ref)

    def test_raises_on_feature_dimension_mismatch(self):
        seq = _make_sequence(50)  # 99 features
        ref = np.random.rand(40, 66)  # wrong number of features
        with pytest.raises(ValueError, match="[Ff]eature dimension"):
            align_sequence(seq, ref)


# ---------------------------------------------------------------------------
# load_reference_template tests
# ---------------------------------------------------------------------------


class TestLoadReferenceTemplate:
    def test_raises_file_not_found_for_unknown_slug(self):
        with pytest.raises(FileNotFoundError):
            load_reference_template("nonexistent_technique_xyz")

    def test_error_message_contains_slug(self):
        slug = "totally_unknown_slug"
        with pytest.raises(FileNotFoundError, match=slug):
            load_reference_template(slug)

    def test_loads_existing_template(self, tmp_path, monkeypatch):
        """Monkeypatch _REFERENCES_DIR so we can write a temp template."""
        import backend.scoring.dtw_aligner as module

        template = _make_sequence(45)
        np.save(tmp_path / "test_kick.npy", template)

        monkeypatch.setattr(module, "_REFERENCES_DIR", tmp_path)

        loaded = load_reference_template("test_kick")
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, template)

    def test_loaded_template_has_expected_shape(self, tmp_path, monkeypatch):
        import backend.scoring.dtw_aligner as module

        template = _make_sequence(60)
        np.save(tmp_path / "front_kick.npy", template)
        monkeypatch.setattr(module, "_REFERENCES_DIR", tmp_path)

        loaded = load_reference_template("front_kick")
        assert loaded.shape == (60, N_FEATURES)

    def test_error_lists_available_templates(self, tmp_path, monkeypatch):
        """Error message should list the slugs that *are* available."""
        import backend.scoring.dtw_aligner as module

        np.save(tmp_path / "front_kick.npy", _make_sequence(30))
        monkeypatch.setattr(module, "_REFERENCES_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="front_kick"):
            load_reference_template("missing_slug")


# ---------------------------------------------------------------------------
# Committed reference templates (LIN-102)
# ---------------------------------------------------------------------------


SUPPORTED_TECHNIQUES = ("front_kick", "roundhouse_kick", "straight_punch")


class TestCommittedReferenceTemplates:
    """Verify that development reference templates ship with the repository."""

    @pytest.mark.parametrize("slug", SUPPORTED_TECHNIQUES)
    def test_template_file_exists(self, slug: str):
        path = _REFERENCES_DIR / f"{slug}.npy"
        assert path.exists(), (
            f"Reference template missing: {path}. "
            f"Run scripts/generate_reference_templates.py to create it."
        )

    @pytest.mark.parametrize("slug", SUPPORTED_TECHNIQUES)
    def test_template_loadable(self, slug: str):
        arr = load_reference_template(slug)
        assert isinstance(arr, np.ndarray), f"{slug}: expected ndarray, got {type(arr)}"

    @pytest.mark.parametrize("slug", SUPPORTED_TECHNIQUES)
    def test_template_shape(self, slug: str):
        arr = load_reference_template(slug)
        assert arr.ndim == 2, f"{slug}: expected 2-D array, got {arr.ndim}-D"
        assert arr.shape[1] == N_FEATURES, (
            f"{slug}: expected {N_FEATURES} features per frame, got {arr.shape[1]}"
        )
        assert arr.shape[0] >= 10, (
            f"{slug}: template has only {arr.shape[0]} frames, expected >= 10"
        )

    @pytest.mark.parametrize("slug", SUPPORTED_TECHNIQUES)
    def test_template_dtype_is_float(self, slug: str):
        arr = load_reference_template(slug)
        assert np.issubdtype(arr.dtype, np.floating), (
            f"{slug}: template dtype {arr.dtype} is not floating-point"
        )

    @pytest.mark.parametrize("slug", SUPPORTED_TECHNIQUES)
    def test_template_has_no_nan_or_inf(self, slug: str):
        arr = load_reference_template(slug)
        assert np.all(np.isfinite(arr)), (
            f"{slug}: template contains NaN or Inf values"
        )


# ---------------------------------------------------------------------------
# align_to_reference tests
# ---------------------------------------------------------------------------


class TestAlignToReference:
    def test_returns_3d_array(self):
        seq = np.tile(
            np.random.default_rng(0).random((33, 3)), (30, 1, 1)
        )
        result = align_to_reference("front_kick", seq)
        assert result.ndim == 3

    def test_output_has_33_landmarks_and_3_coords(self):
        seq = np.tile(
            np.random.default_rng(1).random((33, 3)), (30, 1, 1)
        )
        result = align_to_reference("front_kick", seq)
        assert result.shape[1:] == (33, 3)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            align_to_reference("front_kick", np.zeros((30, 99)))
