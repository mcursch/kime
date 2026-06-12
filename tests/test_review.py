"""
tests/test_review.py

Unit/integration tests for data_pipeline.review.

All filesystem operations are redirected to a temporary directory so the
tests never touch the real data/ tree and run in any CI environment.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_npz(directory: Path, clip_id: str, technique: str, frames: int = 20) -> Path:
    """Write a minimal .npz skeleton file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    landmarks = rng.random((frames, 33, 3), dtype=np.float32)
    path = directory / f"{clip_id}.npz"
    np.savez(path, landmarks=landmarks, technique=technique)
    return path


def _patch_dirs(tmp: Path):
    """Return a dict of patches that redirect all module-level paths."""
    import data_pipeline.review as rv

    return {
        "LANDMARKS_DIR": tmp / "landmarks",
        "TEMPLATES_DIR": tmp / "templates",
        "REJECTED_DIR": tmp / "rejected",
        "PREVIEWS_DIR": tmp / "previews",
        "MANIFEST_PATH": tmp / "templates" / "manifest.json",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManifestIO(unittest.TestCase):
    """Manifest load/save round-trips."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_missing_manifest_returns_empty_list(self):
        import data_pipeline.review as rv

        original = rv.MANIFEST_PATH
        rv.MANIFEST_PATH = self.tmp / "nonexistent" / "manifest.json"
        try:
            result = rv._load_manifest()
            self.assertEqual(result, [])
        finally:
            rv.MANIFEST_PATH = original

    def test_save_and_load_roundtrip(self):
        import data_pipeline.review as rv

        rv.MANIFEST_PATH = self.tmp / "manifest.json"
        self.tmp.mkdir(parents=True, exist_ok=True)
        entries = [{"clip_id": "abc", "technique": "front_kick", "approved_at": "2025-01-01T00:00:00+00:00"}]
        try:
            rv._save_manifest(entries)
            loaded = rv._load_manifest()
            self.assertEqual(loaded, entries)
            # File must be valid JSON
            with rv.MANIFEST_PATH.open() as f:
                json.load(f)
        finally:
            rv.MANIFEST_PATH = Path("data/templates/manifest.json")

    def test_manifest_is_valid_json_after_approve(self):
        """Manifest must be valid JSON immediately after an approve call."""
        import data_pipeline.review as rv

        landmarks_dir = self.tmp / "landmarks"
        templates_dir = self.tmp / "templates"
        manifest_path = templates_dir / "manifest.json"
        npz = _make_npz(landmarks_dir, "clip_001", "roundhouse_kick")

        orig_td = rv.TEMPLATES_DIR
        orig_mp = rv.MANIFEST_PATH
        rv.TEMPLATES_DIR = templates_dir
        rv.MANIFEST_PATH = manifest_path
        try:
            rv._approve(npz, "clip_001", "roundhouse_kick", "tester@example.com")
            with manifest_path.open() as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            entry = data[0]
            self.assertEqual(entry["clip_id"], "clip_001")
            self.assertEqual(entry["technique"], "roundhouse_kick")
            self.assertIn("approved_at", entry)
            # Approved file must exist in templates dir
            self.assertTrue((templates_dir / "roundhouse_kick" / "clip_001.npz").exists())
        finally:
            rv.TEMPLATES_DIR = orig_td
            rv.MANIFEST_PATH = orig_mp


class TestApprove(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _redirect(self, rv):
        rv._orig_td = rv.TEMPLATES_DIR
        rv._orig_mp = rv.MANIFEST_PATH
        rv.TEMPLATES_DIR = self.tmp / "templates"
        rv.MANIFEST_PATH = self.tmp / "templates" / "manifest.json"

    def _restore(self, rv):
        rv.TEMPLATES_DIR = rv._orig_td
        rv.MANIFEST_PATH = rv._orig_mp

    def test_approve_copies_file_to_technique_dir(self):
        import data_pipeline.review as rv

        self._redirect(rv)
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_a", "front_kick")
            rv._approve(npz, "clip_a", "front_kick", "rev@example.com")
            dest = self.tmp / "templates" / "front_kick" / "clip_a.npz"
            self.assertTrue(dest.exists(), f"Expected {dest} to exist")
            # Original must still exist (copy, not move)
            self.assertTrue(npz.exists(), "Source file should still exist after approve")
        finally:
            self._restore(rv)

    def test_approve_updates_manifest_with_required_fields(self):
        import data_pipeline.review as rv

        self._redirect(rv)
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_b", "straight_punch")
            rv._approve(npz, "clip_b", "straight_punch", "rev@example.com")
            entries = rv._load_manifest()
            self.assertEqual(len(entries), 1)
            e = entries[0]
            self.assertIn("approved_at", e)
            self.assertIn("technique", e)
            self.assertEqual(e["technique"], "straight_punch")
            self.assertEqual(e["clip_id"], "clip_b")
            self.assertEqual(e["reviewer_email"], "rev@example.com")
        finally:
            self._restore(rv)

    def test_approve_deduplicates_manifest_by_clip_id(self):
        import data_pipeline.review as rv

        self._redirect(rv)
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_c", "front_kick")
            rv._approve(npz, "clip_c", "front_kick", "a@b.com")
            rv._approve(npz, "clip_c", "front_kick", "a@b.com")
            entries = rv._load_manifest()
            clip_entries = [e for e in entries if e["clip_id"] == "clip_c"]
            self.assertEqual(len(clip_entries), 1, "Duplicate approve should update, not append")
        finally:
            self._restore(rv)


class TestReject(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _redirect(self, rv):
        rv._orig_rd = rv.REJECTED_DIR
        rv.REJECTED_DIR = self.tmp / "rejected"

    def _restore(self, rv):
        rv.REJECTED_DIR = rv._orig_rd

    def test_reject_moves_file_to_rejected_dir(self):
        import data_pipeline.review as rv

        self._redirect(rv)
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_bad", "front_kick")
            rv._reject(npz, "clip_bad", "blurry footage")
            dest = self.tmp / "rejected" / "clip_bad.npz"
            self.assertTrue(dest.exists(), "Rejected file should be in rejected dir")
            self.assertFalse(npz.exists(), "Original should be gone after reject")
        finally:
            self._restore(rv)

    def test_reject_writes_sidecar_with_reason(self):
        import data_pipeline.review as rv

        self._redirect(rv)
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_bad2", "roundhouse_kick")
            rv._reject(npz, "clip_bad2", "occlusion")
            sidecar = self.tmp / "rejected" / "clip_bad2.rejection.json"
            self.assertTrue(sidecar.exists())
            with sidecar.open() as f:
                meta = json.load(f)
            self.assertEqual(meta["reason"], "occlusion")
            self.assertEqual(meta["clip_id"], "clip_bad2")
        finally:
            self._restore(rv)

    def test_reject_does_not_appear_in_templates(self):
        """A rejected file must not end up in templates/, and manifest must not list it."""
        import data_pipeline.review as rv

        templates_dir = self.tmp / "templates"
        rv._orig_rd = rv.REJECTED_DIR
        rv._orig_td = rv.TEMPLATES_DIR
        rv._orig_mp = rv.MANIFEST_PATH
        rv.REJECTED_DIR = self.tmp / "rejected"
        rv.TEMPLATES_DIR = templates_dir
        rv.MANIFEST_PATH = templates_dir / "manifest.json"
        try:
            npz = _make_npz(self.tmp / "landmarks", "clip_bad3", "front_kick")
            rv._reject(npz, "clip_bad3", "bad angle")
            # templates dir should not contain the file
            for p in templates_dir.rglob("clip_bad3.npz"):
                self.fail(f"Rejected file found in templates: {p}")
            # manifest should not contain this clip_id
            entries = rv._load_manifest()
            ids = [e.get("clip_id") for e in entries]
            self.assertNotIn("clip_bad3", ids)
        finally:
            rv.REJECTED_DIR = rv._orig_rd
            rv.TEMPLATES_DIR = rv._orig_td
            rv.MANIFEST_PATH = rv._orig_mp


class TestCollectUnapproved(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_empty_when_landmarks_dir_missing(self):
        import data_pipeline.review as rv

        orig = rv.LANDMARKS_DIR
        rv.LANDMARKS_DIR = self.tmp / "nonexistent"
        try:
            result = rv._collect_unapproved()
            self.assertEqual(result, [])
        finally:
            rv.LANDMARKS_DIR = orig

    def test_excludes_already_approved_clips(self):
        import data_pipeline.review as rv

        landmarks_dir = self.tmp / "landmarks"
        templates_dir = self.tmp / "templates"
        manifest_path = templates_dir / "manifest.json"

        _make_npz(landmarks_dir, "clip_approved", "front_kick")
        _make_npz(landmarks_dir, "clip_pending", "front_kick")

        orig_ld = rv.LANDMARKS_DIR
        orig_td = rv.TEMPLATES_DIR
        orig_mp = rv.MANIFEST_PATH
        rv.LANDMARKS_DIR = landmarks_dir
        rv.TEMPLATES_DIR = templates_dir
        rv.MANIFEST_PATH = manifest_path

        # Pre-approve clip_approved in manifest
        rv._save_manifest([
            {
                "clip_id": "clip_approved",
                "technique": "front_kick",
                "approved_at": "2025-01-01T00:00:00+00:00",
                "reviewer_email": "r@e.com",
            }
        ])
        try:
            unapproved = rv._collect_unapproved()
            stems = [p.stem for p in unapproved]
            self.assertNotIn("clip_approved", stems)
            self.assertIn("clip_pending", stems)
        finally:
            rv.LANDMARKS_DIR = orig_ld
            rv.TEMPLATES_DIR = orig_td
            rv.MANIFEST_PATH = orig_mp


class TestRunReviewIntegration(unittest.TestCase):
    """
    Smoke-test the full run_review() loop with mocked input/rendering.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_with_decisions(self, decisions: list[str], reason: str = "bad"):
        import data_pipeline.review as rv

        landmarks_dir = self.tmp / "landmarks"
        templates_dir = self.tmp / "templates"
        rejected_dir = self.tmp / "rejected"
        manifest_path = templates_dir / "manifest.json"

        npz1 = _make_npz(landmarks_dir, "clip1", "front_kick")
        npz2 = _make_npz(landmarks_dir, "clip2", "roundhouse_kick")

        orig_ld = rv.LANDMARKS_DIR
        orig_td = rv.TEMPLATES_DIR
        orig_rd = rv.REJECTED_DIR
        orig_mp = rv.MANIFEST_PATH
        rv.TEMPLATES_DIR = templates_dir
        rv.REJECTED_DIR = rejected_dir
        rv.MANIFEST_PATH = manifest_path

        # Mock: suppress rendering, feed scripted decisions
        input_values = iter(decisions + [reason])

        def fake_input(_prompt=""):
            return next(input_values, "s")

        with (
            patch.object(rv, "_render_terminal"),
            patch("builtins.input", side_effect=fake_input),
        ):
            rv.run_review(
                reviewer_email="tester@example.com",
                save_mp4=False,
                landmarks_dir=landmarks_dir,
            )

        rv.LANDMARKS_DIR = orig_ld
        rv.TEMPLATES_DIR = orig_td
        rv.REJECTED_DIR = orig_rd
        rv.MANIFEST_PATH = orig_mp

        return templates_dir, rejected_dir, manifest_path

    def test_approve_then_reject(self):
        import data_pipeline.review as rv

        # decisions: approve clip1, then reject clip2 with reason "blurry"
        templates_dir, rejected_dir, manifest_path = self._run_with_decisions(
            ["a", "r", "blurry"]
        )
        # clip1 approved
        self.assertTrue((templates_dir / "front_kick" / "clip1.npz").exists())
        # clip2 rejected
        self.assertTrue((rejected_dir / "clip2.npz").exists())
        # manifest valid
        with manifest_path.open() as f:
            data = json.load(f)
        ids = [e["clip_id"] for e in data]
        self.assertIn("clip1", ids)
        self.assertNotIn("clip2", ids)

    def test_skip_leaves_file_in_place(self):
        import data_pipeline.review as rv

        landmarks_dir = self.tmp / "landmarks2"
        templates_dir = self.tmp / "templates2"
        rejected_dir = self.tmp / "rejected2"
        manifest_path = templates_dir / "manifest.json"
        npz = _make_npz(landmarks_dir, "clip_skip", "straight_punch")

        orig_ld = rv.LANDMARKS_DIR
        orig_td = rv.TEMPLATES_DIR
        orig_rd = rv.REJECTED_DIR
        orig_mp = rv.MANIFEST_PATH
        rv.TEMPLATES_DIR = templates_dir
        rv.REJECTED_DIR = rejected_dir
        rv.MANIFEST_PATH = manifest_path

        with (
            patch.object(rv, "_render_terminal"),
            patch("builtins.input", return_value="s"),
        ):
            rv.run_review(
                reviewer_email="t@e.com",
                save_mp4=False,
                landmarks_dir=landmarks_dir,
            )

        rv.LANDMARKS_DIR = orig_ld
        rv.TEMPLATES_DIR = orig_td
        rv.REJECTED_DIR = orig_rd
        rv.MANIFEST_PATH = orig_mp

        self.assertTrue(npz.exists(), "Skipped file must remain in landmarks dir")
        self.assertFalse((rejected_dir / "clip_skip.npz").exists())

    def test_keyboard_interrupt_exits_gracefully(self):
        import data_pipeline.review as rv

        landmarks_dir = self.tmp / "landmarks3"
        templates_dir = self.tmp / "templates3"
        rejected_dir = self.tmp / "rejected3"
        manifest_path = templates_dir / "manifest.json"
        _make_npz(landmarks_dir, "clip_int", "front_kick")

        orig_ld = rv.LANDMARKS_DIR
        orig_td = rv.TEMPLATES_DIR
        orig_rd = rv.REJECTED_DIR
        orig_mp = rv.MANIFEST_PATH
        rv.TEMPLATES_DIR = templates_dir
        rv.REJECTED_DIR = rejected_dir
        rv.MANIFEST_PATH = manifest_path

        def raise_interrupt(_prompt=""):
            raise KeyboardInterrupt

        with (
            patch.object(rv, "_render_terminal"),
            patch("builtins.input", side_effect=raise_interrupt),
        ):
            # Must not propagate the KeyboardInterrupt
            try:
                rv.run_review(
                    reviewer_email="t@e.com",
                    save_mp4=False,
                    landmarks_dir=landmarks_dir,
                )
            except KeyboardInterrupt:
                self.fail("run_review() should handle KeyboardInterrupt gracefully")

        rv.LANDMARKS_DIR = orig_ld
        rv.TEMPLATES_DIR = orig_td
        rv.REJECTED_DIR = orig_rd
        rv.MANIFEST_PATH = orig_mp


class TestCLI(unittest.TestCase):
    """Smoke-test the CLI argument parsing."""

    def test_review_subcommand_parses(self):
        from data_pipeline.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["review", "--email", "foo@bar.com"])
        self.assertEqual(args.command, "review")
        self.assertEqual(args.reviewer_email, "foo@bar.com")
        self.assertFalse(args.save_mp4)

    def test_mp4_flag(self):
        from data_pipeline.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["review", "--mp4"])
        self.assertTrue(args.save_mp4)

    def test_no_subcommand_prints_help(self):
        from data_pipeline.cli import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm:
            with redirect_stdout(buf):
                main([])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
