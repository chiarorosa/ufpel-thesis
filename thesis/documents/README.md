# Thesis Documentation Hub

This directory is the canonical thesis documentation root.

Use these documents as the authoritative guide for the final defense runtime:

- `baseline-inventory.md`
- `keep-archive-remove-matrix.md`
- `regression-criteria.md`
- `raw-input-contract.md`
- `architecture.md`
- `workflow.md`
- `migration-guide.md`
- `regression-matrix.md`
- `runtime-checklist.md`

Consistency gate:

- Runtime changes in canonical entrypoints (`thesis/scripts/*.py`) MUST be accompanied by aligned updates in this folder.
- `runtime-contract.json` defines the expected documentation/version contract.

Canonical runtime commands:

- `python thesis/scripts/bootstrap_uvg_from_drive.py` (optional dataset bootstrap)
- `python thesis/scripts/clean.py [--execute]`
- `python thesis/scripts/prepare_data.py --run-name <name> ...`
- `python thesis/scripts/train_pipeline.py --run-name <name> ...`
- `python thesis/scripts/evaluate_pipeline.py --run-name <name> ...`
- `python thesis/scripts/run_pipeline_end_to_end.py --run-name <name> ...`
