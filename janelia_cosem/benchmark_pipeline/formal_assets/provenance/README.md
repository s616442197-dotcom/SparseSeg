# Read-only benchmark provenance

This directory preserves source manifests, requested model configs, exact environment snapshots, hardware records, and historical evaluation/timing metadata from the reported runs.

These files are evidence, not executable defaults. Some records intentionally retain the original absolute filesystem paths written at run time. `run_formal_benchmark.py`, `run_benchmark_pipeline.py`, every model adapter, and `evaluation_cross_trials_extreme.py` do not read or resolve files under this directory.

Portable execution inputs live outside this directory:

- fixed masks and their installer: `../paired_roi_masks/`;
- portable MitoNet template: `../configs/mitonet/finetune_config_portable.yaml`;
- SparseSeg adaptive backend: `../sparseseg_adaptive_backend/`;
- model bridges: `../../formal_runners/`.
