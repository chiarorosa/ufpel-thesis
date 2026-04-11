# Regression Acceptance Criteria

## Functional completion criteria

The canonical runtime is accepted when all of the following are true:

1. `thesis/scripts/prepare_data.py` completes and writes manifests plus expected dataset artifacts.
2. `thesis/scripts/train_pipeline.py` completes and produces Stage1, Stage2, Stage3-RECT, Stage3-AB-binary checkpoints.
3. `thesis/scripts/evaluate_pipeline.py` completes and produces pipeline result JSON.
4. `thesis/scripts/clean.py` dry-run and execute modes complete with cleanup manifest output.
5. Contract validators pass:
   - raw input contract (`thesis/uvg/<sequence>/partition_frame_0.txt`)
   - optional binary raw block contract (`thesis/uvg/intra_raw_blocks` with 8/16/32/64 per sequence)
   - docs consistency contract (`thesis/documents/runtime-contract.json`)
6. Canonical runtime path does not select hybrid/ensemble families.
7. Standalone reference guards pass:
   - no `pesquisa_v*` references in thesis-critical runtime/docs paths
   - no numbered `00x_` script exposure in canonical runtime surface/docs

## Metric drift thresholds

Given the existing script-level references and stochastic training behavior:

- Stage2 macro-F1 drift tolerance: ±2.0 pp versus baseline run using equivalent data split and seed.
- Pipeline accuracy drift tolerance: ±1.5 pp versus baseline run using equivalent checkpoints/split.
- Stage3 heads are validated by functional completion first; metric checks are advisory unless thesis committee threshold is specified.

## Pass/fail interpretation

- **Pass**: Functional criteria met and drifts within tolerance.
- **Conditional pass**: Functional criteria met but one metric exceeds tolerance; requires documented explanation.
- **Fail**: Functional criteria not met or multiple metrics exceed tolerance without acceptable rationale.

## Smoke-run exception

For explicit smoke runs (for example, 1 epoch per stage), only functional completion is evaluated. Metric drift thresholds are not applied to smoke evidence.
