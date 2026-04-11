# Baseline Inventory (Standalone Thesis)

## Scope

This baseline captures the standalone thesis command set and expected artifact flow under `thesis/`.

## Baseline Commands

1. Dataset preparation
   - `python thesis/scripts/prepare_dataset.py --base-path <legacy-base> --block-size <8|16|32|64> --output-dir <dir>`
   - Inputs: `<legacy-base>/intra_raw_blocks`, `<legacy-base>/labels`, `<legacy-base>/qps`
   - Outputs: `train.pt`, `val.pt`, `metadata.json`

2. Stage3 dataset preparation
   - `python thesis/scripts/prepare_stage3_datasets.py --base-path <legacy-base> --block-size <size> --output-base <dir>`
   - Inputs: same legacy base path contract
   - Outputs: `RECT/block_<size>/*`, `AB/block_<size>/*`

3. Stage 1 and Stage 2 training (Conv-Adapter solution)
   - `python thesis/scripts/train_adapter_solution.py --dataset-dir <v7_dataset/block_size> --output-dir <dir>`
   - Outputs: `stage1_model_best.pt`, `stage2_adapter_model_best.pt`, metrics/history json/pt

4. Stage3 RECT training
   - `python thesis/scripts/train_stage3_rect.py --dataset-dir <RECT/block_size> --stage2-checkpoint <pt> --output <dir> --fix-batchnorm`
   - Outputs: `model_best.pt`, `metrics.json`, `history.json`

5. Stage3 AB binary training
   - `python thesis/scripts/train_stage3_ab_binary.py --dataset-dir <v7_dataset/block_size> --stage2-checkpoint <pt> --output-dir <dir>`
   - Outputs: `model_best.pt`, `metrics.json`, `history.pt`

6. Hierarchical evaluation (AB binary)
   - `python thesis/scripts/evaluate_pipeline_ab_binary.py --stage1-checkpoint <pt> --stage2-checkpoint <pt> --stage3-rect-checkpoint <pt> --stage3-ab-checkpoint <pt> --dataset-dir <v7_dataset/block_size> --output <dir>`
   - Outputs: `pipeline_<split>_results.json`

## Reference Metrics and Expectations

- Stage2 macro-F1 target zone discussed in code: around 58-60%.
- Pipeline accuracy expectation in AB-binary eval script comments: around 58.5-59.5%.
- Stage3 RECT and AB script comments document expected trends and should be treated as soft reference only.

## Canonicalization Note

The thesis canonical wrappers under `thesis/scripts/` orchestrate this same command family with standardized outputs under `thesis/runs/<run-name>/`.

Canonical semantic entrypoints:

- `thesis/scripts/clean.py`
- `thesis/scripts/prepare_data.py`
- `thesis/scripts/train_pipeline.py`
- `thesis/scripts/evaluate_pipeline.py`
- `thesis/scripts/run_pipeline_end_to_end.py`

When legacy labels/QPs are not available, wrappers can bootstrap a compatible temporary contract from `partition_frame_0.txt` plus `intra_raw_blocks`.
