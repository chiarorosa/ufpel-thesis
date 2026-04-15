# Agent Notes

## What matters most
- This repo's executable source of truth is the canonical runtime in `thesis/runtime/` plus semantic CLIs in `thesis/scripts/`.
- Prefer semantic entrypoints (`prepare_data.py`, `train_pipeline.py`, `evaluate_pipeline.py`, `run_pipeline_end_to_end.py`, `clean.py`); `prepare.py`/`train.py`/`evaluate.py`/`run_end_to_end.py` are compatibility wrappers marked deprecated.
- The only supported runtime family is `conv_adapter_frozen_backbone`; changing it will fail contract checks.

## Canonical command flow
- Run phases in order: `prepare_data.py -> train_pipeline.py -> evaluate_pipeline.py` (or use `run_pipeline_end_to_end.py`).
- Use repo-root execution with explicit interpreter, e.g. `.venv/bin/python thesis/scripts/prepare_data.py ...`.
- `prepare_data.py` requires one of:
  - `--legacy-base-path <path>`, or
  - `--auto-bootstrap-legacy-contract`, or
  - `--skip-legacy-generation`.

## Data-contract gotchas that commonly fail
- Raw input contract is strict: `thesis/uvg/<sequence>/partition_frame_0.txt` must exist.
- `--require-intra-raw-blocks` validates `thesis/uvg/intra_raw_blocks` for all block sizes `8/16/32/64`.
- Default `--min-intra-raw-sequences` is `2`; this repo currently has one sequence under `thesis/uvg/`, so use `--min-intra-raw-sequences 1` for local smoke runs.
- `--auto-generate-intra-raw-blocks` needs source YUV files in `videoset/uvg` and refuses to overwrite existing `*_sample_<block>.txt` files.

## Validation and gates
- `prepare/train/evaluate/end-to-end` run a docs consistency gate by default (`thesis/documents/*` + `runtime-contract.json` + standalone guards). Runtime changes in canonical scripts usually require docs updates too.
- Use `--no-docs-gate` only when intentionally doing runtime-only experimentation.
- Focused checks available:
  - `.venv/bin/python thesis/scripts/validate_standalone.py`
  - `.venv/bin/python thesis/scripts/validate_cleanup_safety.py`
  - `.venv/bin/python thesis/scripts/validate_flow.py --raw-root thesis/uvg --legacy-base-path <path>`

## Directory boundaries
- Treat `thesis/runtime/`, `thesis/scripts/`, `thesis/pipeline/`, `thesis/documents/` as source-of-truth code/docs.
- Treat `thesis/runs/` as generated run artifacts (datasets/checkpoints/evals/manifests); do not edit manually unless task explicitly targets generated outputs.
- Inputs live in `thesis/uvg/` (partition + legacy contract files) and `videoset/uvg/` (YUV videos).

## Cleanup behavior
- `thesis/scripts/clean.py` is dry-run by default; add `--execute` to delete.
- Cleanup is scoped to thesis-generated artifacts and protects `thesis/uvg`, `thesis/runtime`, `thesis/scripts`, `thesis/pipeline`, `thesis/documents`.
