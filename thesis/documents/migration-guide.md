# Migration Guide (Legacy -> Canonical)

## Legacy-to-canonical command mapping

| Legacy intent | Canonical command |
|---|---|
| Dataset preparation (legacy) | `python thesis/scripts/prepare_data.py ...` |
| Stage3 dataset preparation (legacy) | `python thesis/scripts/prepare_data.py ...` |
| Stage training chain | `python thesis/scripts/train_pipeline.py ...` |
| Pipeline evaluation | `python thesis/scripts/evaluate_pipeline.py ...` |
| End-to-end chain | `python thesis/scripts/run_pipeline_end_to_end.py ...` |
| Fresh-start reset | `python thesis/scripts/clean.py [--execute]` |

Compatibility forwarder:

```bash
python thesis/scripts/legacy_forward.py <legacy_command> --run-name <name> ...
```

## Deprecation guidance

1. Prefer `thesis/scripts/*.py` for all defense and thesis-ready runs.
2. Use `legacy_forward.py` only during transition or for script-level continuity.
3. Legacy project folders are optional archival context and are not required by runtime.
4. If legacy `labels/qps` are absent, run canonical prepare with `--auto-bootstrap-legacy-contract`.
5. Canonical surface is semantic and thesis-native (`prepare_data`, `train_pipeline`, `evaluate_pipeline`, `run_pipeline_end_to_end`, `clean`).

## Non-canonical paths

- `thesis/pipeline/ensemble.py` and `thesis/pipeline/hybrid_model.py` (if present in future experiments) are non-default unless explicitly integrated into canonical entrypoints.
