# Regression Matrix

## Matrix definition

| Area | Baseline command(s) | Canonical command | Evidence artifact |
|---|---|---|---|
| Prepare (dataset) | `prepare_dataset`, `prepare_stage3_datasets` | `thesis/scripts/prepare_data.py` | `manifests/prepare.json` |
| Stage1+2 training | `train_adapter_solution` | `thesis/scripts/train_pipeline.py` | `manifests/train.json` |
| Stage3 training | `train_stage3_rect`, `train_stage3_ab_binary` | `thesis/scripts/train_pipeline.py` | `manifests/train.json` |
| Evaluation | `evaluate_pipeline_ab_binary` | `thesis/scripts/evaluate_pipeline.py` | `manifests/evaluate.json` |
| End-to-end | manual chain | `thesis/scripts/run_pipeline_end_to_end.py` | `manifests/end_to_end.json` |
| Fresh start | manual cleanup | `thesis/scripts/clean.py` | `cleanup_reports/*.json` |

## Pass/fail evidence capture

For each run (`thesis/runs/<run-name>`), capture:

1. phase manifests
2. produced artifact paths
3. key metrics json from model/eval outputs
4. pass/fail against `regression-criteria.md`

For auto-bootstrap runs, also capture:

- legacy bootstrap stats in `manifests/prepare.json` (`legacy_bootstrap` section)

## Current session note

Smoke evidence captured in this repository:

- Run: `thesis/runs/e2e_smoke`
- Prepare manifest: `thesis/runs/e2e_smoke/manifests/prepare.json`
- Train manifest: `thesis/runs/e2e_smoke/manifests/train.json`
- Evaluate manifest: `thesis/runs/e2e_smoke/manifests/evaluate.json`
- End-to-end manifest: `thesis/runs/e2e_smoke/manifests/end_to_end.json`
- Pipeline result: `thesis/runs/e2e_smoke/evaluation/pipeline/pipeline_val_results.json`
- Raw-flow validation report: `thesis/runs/e2e_smoke/manifests/flow_validation_manual.json`

Status interpretation:

- Functional end-to-end execution: PASS (prepare, train, evaluate completed).
- Numerical quality: not acceptance-grade because smoke run used 1 epoch for each training phase.
