# Keep / Archive / Remove Matrix

## Scope

This matrix covers canonical runtime artifacts rooted in `thesis/`.

## Scripts and Modules (thesis)

| Path | Decision | Rationale |
|---|---|---|
| `thesis/scripts/prepare_dataset.py` | Keep (internal) | Dataset preparation implementation used by runtime wrappers. |
| `thesis/scripts/prepare_stage3_datasets.py` | Keep (internal) | Stage3 dataset generation implementation. |
| `thesis/scripts/train_adapter_solution.py` | Keep (internal) | Stage1+2 Conv-Adapter training implementation. |
| `thesis/scripts/train_stage3_rect.py` | Keep (internal) | Stage3 RECT training implementation. |
| `thesis/scripts/train_stage3_ab_binary.py` | Keep (internal) | Stage3 AB binary training implementation. |
| `thesis/scripts/evaluate_pipeline_ab_binary.py` | Keep (internal) | Hierarchical evaluation implementation. |
| `thesis/scripts/clean.py` | Keep | Canonical fresh-start cleanup CLI entrypoint. |
| `thesis/scripts/prepare_data.py` | Keep | Canonical prepare-data CLI entrypoint. |
| `thesis/scripts/train_pipeline.py` | Keep | Canonical train-pipeline CLI entrypoint. |
| `thesis/scripts/evaluate_pipeline.py` | Keep | Canonical evaluate-pipeline CLI entrypoint. |
| `thesis/scripts/run_pipeline_end_to_end.py` | Keep | Canonical end-to-end CLI entrypoint. |
| `thesis/scripts/prepare.py` | Archive (compatibility) | Transitional wrapper; not canonical. |
| `thesis/scripts/train.py` | Archive (compatibility) | Transitional wrapper; not canonical. |
| `thesis/scripts/evaluate.py` | Archive (compatibility) | Transitional wrapper; not canonical. |
| `thesis/scripts/run_end_to_end.py` | Archive (compatibility) | Transitional wrapper; not canonical. |
| `thesis/scripts/legacy_forward.py` | Keep (transition) | Optional compatibility forwarder during migration. |
| `thesis/runtime/canonical.py` | Keep | Canonical orchestrator used by entrypoints. |
| `thesis/runtime/contracts.py` | Keep | Runtime/docs/flow contract validation and handoff guards. |
| `thesis/runtime/legacy_contract.py` | Keep | Bootstrap legacy labels/QPs from partition contract. |
| `thesis/runtime/flow_validation.py` | Keep | Raw-flow validator (`partition -> intra_raw_blocks -> labels/qps`). |
| `thesis/pipeline/backbone.py` | Keep | Core backbone used by canonical training/evaluation scripts. |
| `thesis/pipeline/conv_adapter.py` | Keep | Conv-Adapter implementation and frozen-backbone behavior. |
| `thesis/pipeline/data_hub.py` | Keep | Dataset loading/mapping contract utilities. |
| `thesis/pipeline/evaluation.py` | Keep | Shared evaluation/metrics utilities. |
| `thesis/pipeline/losses.py` | Keep | Core loss definitions used in canonical training flow. |

## External legacy folders

- `pesquisa_v*` trees are out of scope for canonical runtime and may be archived or removed without breaking `thesis/` execution.
