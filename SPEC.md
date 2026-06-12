# Kime

**Kime** (決め) — the focused snap of power at the instant a strike lands. This app judges whether your technique has it.

Kime is a web app that analyzes video of a user performing martial arts techniques, scores the technique's accuracy and correctness, and explains *why* it was good or bad in plain coaching language. Results are viewable in a dashboard with per-criterion breakdowns, skeleton-overlay playback, and progress history.

## MVP scope

Striking basics, judged from a single camera:

- Front kick
- Roundhouse kick
- Straight punch

## How judging works

The pipeline is pose-based and interpretable — biomechanics produce the numbers, a VLM produces the prose:

1. **Pose extraction** — MediaPipe Pose Landmarker extracts 33 3D landmarks per frame (pixel + metric world coordinates).
2. **Normalization** — skeletons are hip-centered, scaled by torso length, rotated to a canonical facing direction, and smoothed (Savitzky–Golay), so any body type and camera placement compares fairly against the reference.
3. **Segmentation** — the technique's execution window (chamber → extension → retraction) is located via joint-velocity peaks.
4. **Alignment** — Dynamic Time Warping matches the user's motion phase-by-phase to expert reference templates, so slower execution isn't penalized as bad form.
5. **Scoring** — rule-based biomechanical criteria computed on the aligned skeletons: chamber height, hip rotation at impact, extension angle, balance (center of mass over the support foot), guard position, retraction speed. Each criterion yields a score plus its raw numeric delta from the reference.
6. **Feedback** — the metric deltas and annotated keyframes go to the Claude API, which writes grounded coaching feedback (e.g., "your hip stays square at impact — the reference rotates ~40° more, which is where roundhouse power comes from").

## Reference data

An offline pipeline builds the expert reference library: yt-dlp scrapes publicly viewable instructional and competition clips per technique, filters to clean single-person shots, extracts poses, and stores only the **skeleton sequences** as reference templates — landmarks are kept, raw video is not.

## Architecture

- **Backend / ML** — Python (WSL): FastAPI, MediaPipe, OpenCV, NumPy/SciPy, `dtaidistance` (DTW), yt-dlp + ffmpeg (scraping), Anthropic SDK (feedback), SQLite (uploads, scores, history). Analysis runs as async jobs — upload returns immediately and the dashboard polls until results are ready.
- **Frontend** — React + Vite dashboard: upload/record page, per-attempt results with overall score, per-criterion radar chart, video playback with skeleton overlay, and progress over time.

## Development plan (agentic workflow)

The build runs in six phases, each ending with a verification gate that must pass before the next phase starts. Phases are orchestrated sequentially, with parallel subagents only in phase 5 where backend and frontend are independent.

| Phase | Deliverable | Verification gate |
|---|---|---|
| 1. Scaffold | Repo structure, WSL Python env, deps installed, git init | MediaPipe processes a test video end-to-end |
| 2. Data agent | Scraper + clip filter + reference template builder for 3 techniques | ≥10 clean reference templates per technique, visually spot-checked |
| 3. Vision module | Pose extraction, normalization, smoothing, segmentation, with unit tests | Segmentation finds the rep window on held-out clips |
| 4. Judging engine | DTW alignment, criterion metrics, scoring rules | Expert clips score high, deliberately-bad clips score low (sanity eval report) |
| 5. App | FastAPI API + React dashboard, skeleton-overlay renderer | Upload → score → feedback works in the browser |
| 6. End-to-end verify | Full run on fresh videos, README, eval summary | Usable by an end user |

Phase 2 includes a human review step: candidate reference clips' skeletons are shown for approval before becoming gold standards. Phases 1–4 require no API key; the Claude feedback step (phase 5+) needs `ANTHROPIC_API_KEY`.

## Status

Pre-development. Architecture and scope confirmed; implementation not yet started.

## Known limitations

- Single-camera judging is sensitive to filming angle — the app detects facing direction and warns on bad angles rather than silently mis-scoring.
- Reference template quality depends on scraped footage; candidate references are human-reviewed before becoming gold standards.
