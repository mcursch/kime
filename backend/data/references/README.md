# Reference Templates

This directory holds per-technique DTW reference templates used by the
`backend/scoring/dtw_aligner` module.

## File naming

Each template file is named `<technique_slug>.npy`, where `technique_slug`
matches the identifier passed to `load_reference_template()`.

Currently supported technique slugs:

| Slug | Description |
|------|-------------|
| `front_kick` | Front kick (mae geri) |
| `roundhouse_kick` | Roundhouse kick (mawashi geri) |
| `straight_punch` | Straight punch (choku tsuki / jab) |

## Array format

Templates are NumPy arrays saved with `np.save()` and have the following shape:

```
(n_frames, n_features)
```

where:

- **`n_frames`** — number of frames in the reference recording.  The DTW
  aligner handles variable-length sequences, so the template length does not
  need to match the query length.
- **`n_features`** — number of landmark features per frame.  For the
  MediaPipe Pose model this is `33 landmarks × 3 coordinates = 99`.

### Coordinate convention

Features are laid out as a flat row vector in MediaPipe landmark order:

```
[x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]
```

Sequences must be **normalized** (hip-centred, unit-torso-height scale) and
**smoothed** (e.g. Savitzky-Golay) before being saved as reference templates
or passed to `align_sequence()`.

## Generating a template

1. Record a clean representative repetition with MediaPipe Pose.
2. Run the normalization + smoothing pipeline to produce a
   `(n_frames, 99)` float64 array.
3. Segment the rep window to isolate a single repetition.
4. Save with `np.save("backend/data/references/<slug>.npy", array)`.

> **Note:** This directory is checked in without `.npy` files.  Actual
> template files are generated offline from expert recordings and are not
> committed to the repository due to their size.  Add them locally before
> running the scoring pipeline.
