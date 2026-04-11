# Final Thesis Runtime Checklist

## Accepted command set (defense usage)

1. `python thesis/scripts/clean.py [--execute]`
2. `python thesis/scripts/prepare_data.py --run-name <name> ...`
3. `python thesis/scripts/train_pipeline.py --run-name <name> ...`
4. `python thesis/scripts/evaluate_pipeline.py --run-name <name> ...`
5. `python thesis/scripts/run_pipeline_end_to_end.py --run-name <name> ...`

## Readiness checks

- [x] Raw input contract validated (`thesis/uvg/<sequence>/partition_frame_0.txt`)
- [x] Optional intra raw blocks validated (`thesis/uvg/intra_raw_blocks`, 8/16/32/64)
- [x] Legacy contract resolved (`--legacy-base-path` or `--auto-bootstrap-legacy-contract`)
- [x] Prepare handoff artifacts present
- [x] Train checkpoints present (Stage1/2/3)
- [x] Evaluation result JSON generated
- [x] Conv-Adapter + frozen-backbone static gate passed
- [x] Docs consistency gate passed (`runtime-contract.json` + required docs)
- [ ] Regression criteria reviewed and pass/fail recorded

## Documentation links

- `thesis/documents/architecture.md`
- `thesis/documents/workflow.md`
- `thesis/documents/migration-guide.md`
- `thesis/documents/regression-criteria.md`
- `thesis/documents/regression-matrix.md`
