# Canonical Thesis Architecture

## Runtime boundaries

- Canonical orchestration package: `thesis/runtime/`
  - `contracts.py`: path contracts, handoff validators, docs gate, runtime-family checks.
  - `runner.py`: subprocess execution helper.
  - `canonical.py`: prepare/train/evaluate/end-to-end orchestration.
  - `legacy_contract.py`: temporary legacy labels/qps bootstrap from partition contract.

- Canonical CLI entrypoints: `thesis/scripts/`
  - `clean.py`
  - `prepare_data.py`
  - `train_pipeline.py`
  - `evaluate_pipeline.py`
  - `run_pipeline_end_to_end.py`

- Internal implementation scripts (non-canonical surface):
  - `prepare_dataset.py`
  - `prepare_stage3_datasets.py`
  - `train_adapter_solution.py`
  - `train_stage3_rect.py`
  - `train_stage3_ab_binary.py`
  - `evaluate_pipeline_ab_binary.py`

## Model family rule (mandatory)

The only allowed canonical family is:

- `conv_adapter_frozen_backbone`

Implications:

1. Stage2 uses Conv-Adapter (`AdapterBackbone`) over frozen backbone.
2. Stage3 RECT and Stage3 AB binary run with frozen backbones sourced from Stage2.
3. Hybrid and ensemble modules remain outside default runtime path.

## Data and artifact boundaries

- Raw contract root: `thesis/uvg`
- Optional binary raw-block contract root: `thesis/uvg/intra_raw_blocks`
- Canonical run root: `thesis/runs/<run-name>/`
  - `datasets/`
  - `training/`
  - `evaluation/`
  - `manifests/`
