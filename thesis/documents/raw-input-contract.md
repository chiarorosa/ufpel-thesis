# Raw Input Contract

## Canonical contract (mandatory)

The thesis runtime is rooted in:

- `thesis/uvg/<sequence>/partition_frame_0.txt`

Rules:

1. `<sequence>` is a directory under `thesis/uvg`.
2. `partition_frame_0.txt` MUST exist for each required sequence.
3. Missing files fail fast with actionable error containing missing paths.

## Binary raw blocks contract (current repository context)

The repository currently includes binary raw blocks at:

- `thesis/uvg/intra_raw_blocks/<sequence>_sample_<block>.txt`

Where `<block>` is one of `8`, `16`, `32`, `64`.

Current confirmed sequences with full 4-size coverage:

- `Beauty_3840x2160_120fps_420_10bit`

Validation behavior:

- When `--require-intra-raw-blocks` is enabled, canonical prepare enforces at least 2 sequences with complete 8/16/32/64 coverage by default.

### Canonical generation mode (no XLSX)

When `intra_raw_blocks` files are missing, canonical prepare can generate them directly from:

- `thesis/uvg/<sequence>/partition_frame_0.txt`
- `videoset/uvg/<sequence>.yuv` (or custom `--videos-root`)

Enable with:

- `--auto-generate-intra-raw-blocks`

Generation rules (rearrange-compatible only):

1. Reads Y plane from YUV 4:2:0 10-bit little-endian.
2. Uses frame number from `partition_frame_<n>.txt` (default `partition_frame_0.txt`).
3. For each block size (`8/16/32/64`), sorts partition entries by `(row, col, order_hint)`.
4. Builds row-major candidate blocks on padded frame grid (same strategy as primary rearrange).
5. Applies legacy-compatible selection by column index (`lcols`) derived from partition `col`.
6. Writes `thesis/uvg/intra_raw_blocks/<sequence>_sample_<8|16|32|64>.txt`.
7. Never overwrites existing sample files (fails fast if targets exist).

Coordinate interpretation:

- `row`/`col` are in AV1 4x4 units.
- Selection index mapping follows rearrange rule:
  `lcol = (col_u4 / block_size) * partition_coord_scale`.
- Default `partition_coord_scale` is `4`.

## Legacy base-path compatibility

The baseline dataset scripts (`001`, `002`) still require legacy layout:

- `<base>/intra_raw_blocks`
- `<base>/labels`
- `<base>/qps`

Canonical wrappers expose this through `--legacy-base-path` to preserve compatibility while maintaining thesis contracts.

### Auto-bootstrap mode

When legacy labels/QPs are unavailable, canonical prepare supports:

- `--auto-bootstrap-legacy-contract`

Behavior:

1. Copies `thesis/uvg/intra_raw_blocks/*_sample_<8|16|32|64>.txt` into a temporary legacy contract root under the run directory.
2. Derives labels and QPs from `thesis/uvg/<sequence>/partition_frame_0.txt`.
3. Writes:
   - `labels/<sequence>_labels_<block>_intra.txt`
   - `qps/<sequence>_qps_<block>_intra.txt`

Compatibility detail:

- Canonical bootstrap now uses `rearrange_exact` alignment only.
- Labels/QPs are generated with the same rearrange-compatible ordering used by samples.
- Drop-first and heuristic auto-alignment are not used in the new flow.

Flow validator:

- `thesis/scripts/validate_flow.py` checks end-to-end consistency for each active sequence/block:
  - `partition_frame_0.txt` entries (intra + block_size)
  - `intra_raw_blocks/<sequence>_sample_<block>.txt` block count
  - `labels/<sequence>_labels_<block>_intra.txt` count
  - `qps/<sequence>_qps_<block>_intra.txt` count

## Visual sample generation (optional)

Canonical prepare can generate JPEG previews after legacy contract is available:

- Enable with `--generate-visual-samples`
- Default output root: `thesis/runs/<run-name>/artifacts/visual_samples`

Per file naming convention:

- `<sequence>_<block_size>_<class_name>_<qp>.<ext>`

Where `<class_name>` uses canonical partition names (e.g., `PARTITION_SPLIT`,
`PARTITION_HORZ_A`, `PARTITION_VERT`).

Behavior:

1. Loads samples from `intra_raw_blocks` and aligned labels/QPs from legacy contract.
2. Converts 10-bit pixels (`0..1023`) to display range (`0..255`).
3. Selects center-ROI samples per `(label, qp)` pair using partition
   coordinates (closest candidates to frame-center region, instead of first-occurrence only).
4. Saves up to `--visual-max-per-label-qp` images per `(label, qp)` pair.
4. Uses nearest-neighbor visualization scaling via `--visual-scale`.

## Full-frame overlay generation (optional)

Canonical prepare can generate one consolidated full-frame overlay image:

- Enable with `--generate-frame-overlay`
- Optional frame selector: `--overlay-frame-number` (default `0`)
- Default output root: `thesis/runs/<run-name>/artifacts/frame_overlays`

Output naming:

- `<sequence>_frame<n>_overlay_all_blocks.<ext>`

Behavior:

1. Loads Y plane from `<videos-root>/<sequence>.yuv` for frame `n`.
2. Loads `thesis/uvg/<sequence>/partition_frame_<n>.txt`.
3. Builds a leaf-tiling partition view (non-overlapping block partition) from
   non-SPLIT candidates.
4. Fills any residual uncovered units with `PARTITION_GAP_FILL` to guarantee
   complete frame coverage.
5. Applies class-color overlay without text inside blocks.
6. Adds legend with color and count per class.

Coordinate interpretation matches extraction mode via `--partition-coord-scale`
(default `4`).

Image format:

- `jpg/jpeg`, `png`, `tif/tiff`
- For near-zero compression validation, prefer `png` (recommended) or `tiff`.
