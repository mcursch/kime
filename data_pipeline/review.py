"""
data_pipeline/review.py

Human-review CLI for approving candidate skeleton files (.npz) as reference
templates.  Invoked via ``python -m data_pipeline.cli review``.

Workflow for each unapproved file in data/landmarks/
  1. Render a stick-figure animation (terminal ASCII or a saved MP4 preview).
  2. Prompt the operator: [a]pprove / [r]eject / [s]kip.
  3. Approved  → copied to data/templates/<technique>/, manifest.json updated.
     Rejected  → moved to data/rejected_landmarks/ with a reason string.
     Skipped   → left in place, revisited on next run.
  4. Keyboard-interrupt (Ctrl-C) exits cleanly after the current file is done.

manifest.json is written atomically (temp-file + rename) so it is never
partially written.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Directory layout (all relative to the repository root)
# ---------------------------------------------------------------------------

LANDMARKS_DIR = Path("data/landmarks")
TEMPLATES_DIR = Path("data/templates")
REJECTED_DIR = Path("data/rejected_landmarks")
PREVIEWS_DIR = Path("data/review_previews")
MANIFEST_PATH = TEMPLATES_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# MediaPipe 33-landmark skeleton connectivity
# (pairs of landmark indices that form "bones")
# ---------------------------------------------------------------------------

SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # face / head (nose to eyes/ears)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> list[dict]:
    """Return the current manifest list, or [] if not yet created."""
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    return []


def _save_manifest(entries: list[dict]) -> None:
    """Atomically write the manifest so it is never partially updated."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=MANIFEST_PATH.parent, prefix=".manifest_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, MANIFEST_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _clip_id_from_path(npz_path: Path) -> str:
    """Use the stem of the filename as the clip ID."""
    return npz_path.stem


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------


def _load_landmarks(npz_path: Path) -> tuple[np.ndarray, str]:
    """
    Load landmark array and technique label from an .npz file.

    Expected keys in the archive:
      landmarks : float32 array of shape (T, 33, 3) or (T, 33, 4)
      technique : scalar string  (e.g. "front_kick")

    Falls back gracefully if the 'technique' key is absent.
    """
    data = np.load(npz_path, allow_pickle=False)
    landmarks: np.ndarray = data["landmarks"]  # (T, 33, >=3)
    technique: str = str(data["technique"]) if "technique" in data else "unknown"
    return landmarks, technique


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _ascii_frame(landmarks_xy: np.ndarray, width: int = 60, height: int = 24) -> str:
    """
    Render a single frame of 2-D landmarks as ASCII art.

    landmarks_xy : (33, 2) array of (x, y) coordinates in [0, 1] range.
    """
    canvas = [[" "] * width for _ in range(height)]

    def _plot(col: int, row: int, ch: str = "o") -> None:
        r = max(0, min(height - 1, row))
        c = max(0, min(width - 1, col))
        canvas[r][c] = ch

    # Draw joints
    for i, (x, y) in enumerate(landmarks_xy):
        col = int(x * (width - 1))
        row = int(y * (height - 1))
        _plot(col, row, "o")

    # Draw bones (Bresenham-style via numpy)
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(landmarks_xy) or b >= len(landmarks_xy):
            continue
        x0, y0 = landmarks_xy[a]
        x1, y1 = landmarks_xy[b]
        c0, r0 = int(x0 * (width - 1)), int(y0 * (height - 1))
        c1, r1 = int(x1 * (width - 1)), int(y1 * (height - 1))
        steps = max(abs(c1 - c0), abs(r1 - r0), 1)
        for step in range(steps + 1):
            t = step / steps
            c = int(c0 + t * (c1 - c0))
            r = int(r0 + t * (r1 - r0))
            if canvas[max(0, min(height - 1, r))][max(0, min(width - 1, c))] == " ":
                _plot(c, r, ".")

    return "\n".join("".join(row) for row in canvas)


def _render_terminal(landmarks: np.ndarray, clip_id: str, technique: str) -> None:
    """
    Animate skeleton frames in the terminal by repeatedly printing ASCII art.

    Shows up to 60 frames, cycling if the clip is longer, then pauses.
    """
    T = landmarks.shape[0]
    # Project to 2-D: use x and y (world coordinates index 0 and 1).
    # Landmarks may be (T,33,3) or (T,33,4); we use columns 0 (x) and 1 (y).
    xy = landmarks[:, :, :2]  # (T, 33, 2)

    # Normalise so values fall in [0, 1] for display.
    x_min, x_max = xy[:, :, 0].min(), xy[:, :, 0].max()
    y_min, y_max = xy[:, :, 1].min(), xy[:, :, 1].max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    display_frames = min(T, 60)
    indices = np.linspace(0, T - 1, display_frames, dtype=int)

    print(f"\n  Clip: {clip_id}  |  Technique: {technique}  |  Frames: {T}")
    print("  " + "-" * 60)

    import time

    for idx in indices:
        frame_xy = xy[idx].copy()
        frame_xy[:, 0] = (frame_xy[:, 0] - x_min) / x_range
        # Flip y so that head is at top
        frame_xy[:, 1] = 1.0 - (frame_xy[:, 1] - y_min) / y_range
        art = _ascii_frame(frame_xy, width=60, height=22)
        # Move cursor up to overwrite previous frame (ANSI escape)
        sys.stdout.write("\033[24A" if idx != indices[0] else "")
        sys.stdout.write(art + "\n")
        sys.stdout.flush()
        time.sleep(0.08)

    print("  " + "-" * 60)


def _render_mp4(landmarks: np.ndarray, clip_id: str, technique: str) -> Path:
    """
    Save a stick-figure animation as an MP4 in data/review_previews/.

    Returns the path to the saved file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEWS_DIR / f"{clip_id}.mp4"

    T = landmarks.shape[0]
    xy = landmarks[:, :, :2]
    x_min, x_max = xy[:, :, 0].min(), xy[:, :, 0].max()
    y_min, y_max = xy[:, :, 1].min(), xy[:, :, 1].max()
    margin = 0.05
    x_pad = (x_max - x_min) * margin or 0.1
    y_pad = (y_max - y_min) * margin or 0.1

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)  # invert y so head is up
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{clip_id} — {technique}", fontsize=10)

    joint_scatter = ax.scatter([], [], s=20, c="steelblue", zorder=3)
    bone_lines = [ax.plot([], [], "gray", lw=1)[0] for _ in SKELETON_CONNECTIONS]

    def _update(frame_idx: int):
        frame_xy = xy[frame_idx]
        joint_scatter.set_offsets(frame_xy)
        for line, (a, b) in zip(bone_lines, SKELETON_CONNECTIONS):
            if a < len(frame_xy) and b < len(frame_xy):
                xs = [frame_xy[a, 0], frame_xy[b, 0]]
                ys = [frame_xy[a, 1], frame_xy[b, 1]]
                line.set_data(xs, ys)
        return [joint_scatter] + bone_lines

    ani = animation.FuncAnimation(
        fig, _update, frames=T, interval=50, blit=True
    )

    writer = animation.FFMpegWriter(fps=20, bitrate=800)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Approve / reject actions
# ---------------------------------------------------------------------------


def _approve(
    npz_path: Path,
    clip_id: str,
    technique: str,
    reviewer_email: str,
) -> None:
    """Copy the skeleton to templates/<technique>/ and update manifest.json."""
    dest_dir = TEMPLATES_DIR / technique
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / npz_path.name
    shutil.copy2(npz_path, dest_path)

    entries = _load_manifest()
    # Remove any pre-existing entry for this clip_id
    entries = [e for e in entries if e.get("clip_id") != clip_id]
    entries.append(
        {
            "clip_id": clip_id,
            "technique": technique,
            "source_file": npz_path.name,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "reviewer_email": reviewer_email,
        }
    )
    _save_manifest(entries)
    print(f"  ✓  Approved → {dest_path}")


def _reject(npz_path: Path, clip_id: str, reason: str) -> None:
    """Move the skeleton to data/rejected_landmarks/ with a sidecar reason file."""
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = REJECTED_DIR / npz_path.name
    shutil.move(str(npz_path), dest_path)

    # Write a small JSON sidecar so the reason is recorded alongside the file
    sidecar = dest_path.with_suffix(".rejection.json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_id": clip_id,
                "rejected_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"  ✗  Rejected → {dest_path}")


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def _collect_unapproved() -> list[Path]:
    """
    Return .npz files in data/landmarks/ that are not yet in the manifest.
    """
    if not LANDMARKS_DIR.exists():
        return []
    approved_ids: set[str] = {
        e["clip_id"] for e in _load_manifest() if "clip_id" in e
    }
    candidates = sorted(LANDMARKS_DIR.glob("*.npz"))
    return [p for p in candidates if _clip_id_from_path(p) not in approved_ids]


def run_review(
    reviewer_email: str,
    save_mp4: bool = False,
    landmarks_dir: Optional[Path] = None,
) -> None:
    """
    Entry point called by the CLI.

    Parameters
    ----------
    reviewer_email:
        E-mail address recorded in manifest entries.
    save_mp4:
        If True, save an MP4 preview instead of rendering in the terminal.
    landmarks_dir:
        Override the default LANDMARKS_DIR (useful for tests).
    """
    global LANDMARKS_DIR
    if landmarks_dir is not None:
        LANDMARKS_DIR = Path(landmarks_dir)

    candidates = _collect_unapproved()

    if not candidates:
        print("No unapproved skeleton files found in", LANDMARKS_DIR)
        return

    print(f"\nFound {len(candidates)} unapproved skeleton(s) to review.\n")

    interrupted = False
    for i, npz_path in enumerate(candidates, start=1):
        print(f"\n[{i}/{len(candidates)}] {npz_path.name}")

        try:
            landmarks, technique = _load_landmarks(npz_path)
        except Exception as exc:
            print(f"  ! Could not load {npz_path.name}: {exc} — skipping.")
            continue

        clip_id = _clip_id_from_path(npz_path)

        # --- Render preview ---
        if save_mp4:
            try:
                preview_path = _render_mp4(landmarks, clip_id, technique)
                print(f"  Preview saved: {preview_path}")
            except Exception as exc:
                print(f"  ! MP4 render failed ({exc}); falling back to terminal.")
                save_mp4 = False  # don't try again for subsequent files

        if not save_mp4:
            try:
                _render_terminal(landmarks, clip_id, technique)
            except Exception as exc:
                print(f"  ! Terminal render failed: {exc}")

        # --- Prompt ---
        try:
            decision = _prompt_decision(clip_id, technique)
        except KeyboardInterrupt:
            print("\n\nInterrupted — exiting review session.")
            interrupted = True
            break

        if decision == "approve":
            try:
                _approve(npz_path, clip_id, technique, reviewer_email)
            except Exception as exc:
                print(f"  ! Approve failed: {exc}")
        elif decision == "reject":
            reason = _prompt_reason()
            try:
                _reject(npz_path, clip_id, reason)
            except Exception as exc:
                print(f"  ! Reject failed: {exc}")
        else:
            print("  →  Skipped.")

    if not interrupted:
        print("\nReview session complete.")


def _prompt_decision(clip_id: str, technique: str) -> str:
    """
    Ask the operator what to do.

    Returns one of "approve", "reject", "skip".
    Loops until a valid input is given (KeyboardInterrupt propagates up).
    """
    prompt = (
        f"\n  Clip: {clip_id!r}  |  Technique: {technique!r}\n"
        "  [a] Approve   [r] Reject   [s] Skip\n"
        "  Decision: "
    )
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            return "skip"
        if raw in ("a", "approve"):
            return "approve"
        if raw in ("r", "reject"):
            return "reject"
        if raw in ("s", "skip", ""):
            return "skip"
        print("  Please enter 'a', 'r', or 's'.")


def _prompt_reason() -> str:
    """Ask for a rejection reason; returns a default if empty."""
    prompt = "  Rejection reason (press Enter to skip): "
    try:
        reason = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        reason = ""
    return reason or "No reason given"
