# Canonical Workflow

## 1) Prepare

### 0) (Opcional) Bootstrap de dados UVG via Google Drive

```bash
python thesis/scripts/bootstrap_uvg_from_drive.py
```

Opções comuns:

- `--drive-url <url>` para informar um espelho alternativo.
- `--overwrite` para sobrescrever arquivos já existentes em `thesis/uvg`.
- `--strict-download` para abortar se qualquer arquivo do Drive falhar.
- `--dry-run` para apenas planejar sync sem download/movimentação.

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --legacy-base-path <legacy-base-path>
```

For contract-only validation (no legacy generation):

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --skip-legacy-generation \
  --require-intra-raw-blocks
```

When `labels/qps` are missing but `intra_raw_blocks` + partition files exist:

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --raw-root thesis/uvg \
  --require-intra-raw-blocks \
  --auto-bootstrap-legacy-contract
```

When `intra_raw_blocks` are also missing, generate them directly from
`partition_frame_0.txt` + source YUVs (no XLSX required):

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --raw-root thesis/uvg \
  --videos-root videoset/uvg \
  --video-ext yuv \
  --width 3840 \
  --height 2160 \
  --auto-generate-intra-raw-blocks \
  --require-intra-raw-blocks \
  --min-intra-raw-sequences 1 \
  --auto-bootstrap-legacy-contract
```

Notes:

- Generated files are written to `thesis/uvg/intra_raw_blocks`.
- Existing `*_sample_<8|16|32|64>.txt` files are never overwritten (fail-fast).
- Extraction strategy is rearrange-compatible (primary-method parity) for 8/16/32/64.
- Label/QP bootstrap uses rearrange-exact ordering (same order as extracted samples).

Generate visual JPEG previews directly from extracted binaries:

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --raw-root thesis/uvg \
  --require-intra-raw-blocks \
  --min-intra-raw-sequences 1 \
  --auto-bootstrap-legacy-contract \
  --generate-visual-samples \
  --visual-max-per-label-qp 1 \
  --visual-scale 8
```

Visual output default:

- `thesis/runs/<run-name>/artifacts/visual_samples/<sequence>/block_<size>/`
- Naming pattern: `<sequence>_<block_size>_<class_name>_<qp>.jpg`

Generate one consolidated full-frame overlay in leaf-tiling partition mode:

```bash
python thesis/scripts/prepare_data.py \
  --run-name defense_r1 \
  --raw-root thesis/uvg \
  --videos-root videoset/uvg \
  --width 3840 \
  --height 2160 \
  --generate-frame-overlay \
  --overlay-frame-number 0 \
  --overlay-image-format png
```

Frame overlay output:

- `thesis/runs/<run-name>/artifacts/frame_overlays/<sequence>/`
- `<sequence>_frame<n>_overlay_all_blocks.<ext>`

The image includes a color legend and class counts, with no text drawn inside blocks.
Overlay uses leaf-tiling (non-overlapping block partition view), not all decision nodes.
Use `png` (recommended) or `tiff` for near-lossless inspection.

## 2) Train

```bash
python thesis/scripts/train_pipeline.py \
  --run-name defense_r1 \
  --block-size 16 \
  --device cuda
```

## 3) Evaluate

```bash
python thesis/scripts/evaluate_pipeline.py \
  --run-name defense_r1 \
  --block-size 16 \
  --split val \
  --device cuda
```

## 4) End-to-end

```bash
python thesis/scripts/run_pipeline_end_to_end.py \
  --run-name defense_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --legacy-base-path <legacy-base-path> \
  --device cuda \
  --split val
```

Quick smoke run (functional verification only):

```bash
python thesis/scripts/run_pipeline_end_to_end.py \
  --run-name e2e_smoke \
  --raw-root thesis/uvg \
  --videos-root videoset/uvg \
  --auto-generate-intra-raw-blocks \
  --require-intra-raw-blocks \
  --min-intra-raw-sequences 1 \
  --auto-bootstrap-legacy-contract \
  --python .venv/bin/python \
  --device cpu \
  --epochs-stage2 1 \
  --epochs-stage3-rect 1 \
  --epochs-stage3-ab-binary 1 \
  --batch-size 256 \
  --split val
```

## 5) Fresh start cleanup

Preview cleanup targets (safe dry-run):

```bash
python thesis/scripts/clean.py
```

Execute cleanup of generated artifacts:

```bash
python thesis/scripts/clean.py --execute
```

Cleanup only one run:

```bash
python thesis/scripts/clean.py --run-name <run-name> --execute
```

Cleanup safety guarantees:

- Never deletes source input contract under `thesis/uvg/**`.
- Never deletes thesis source/docs under `thesis/runtime/**`, `thesis/scripts/**`, `thesis/pipeline/**`, `thesis/documents/**`.
- Always writes a cleanup manifest report.

## Handoff validation

- Prepare validates dataset artifacts needed by training.
- Train validates checkpoint artifacts needed by evaluation.
- Each phase writes a manifest in `thesis/runs/<run-name>/manifests/`.

## Raw-flow validation

You can validate the raw contract chain explicitly:

```bash
python thesis/scripts/validate_flow.py \
  --raw-root thesis/uvg \
  --legacy-base-path thesis/runs/<run-name>/datasets/legacy_bootstrap \
  --output-json thesis/runs/<run-name>/manifests/flow_validation_manual.json
```
