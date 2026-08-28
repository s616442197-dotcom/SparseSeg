# 12-model benchmark reproduction pipeline

This folder is the maintained Reviewer 3 reproduction interface. It supports three workflows:

1. `run_formal_benchmark.py` expands and runs the formal 12-model x 15-case grid.
2. `run_benchmark_pipeline.py --epochs 1` runs the same 12 adapters on the packaged small EM example as an interface check. These values are not manuscript results.
3. `run_benchmark_pipeline.py --plots-only` redraws both reviewer figures from the released CSV tables without model environments or raw data.

## Released contents

- `data/benchmark_metrics.csv`: 12 models x 5 seeds x 3 ROI budgets (180 rows).
- `data/time_per_epoch.csv`: per-epoch timings used by `Fig_add1_v3.pdf`.
- `data/wallclock_total.csv`: total end-to-end wall-clock values used by `Fig_add_more1_v3.pdf`.
- `outputs/`: the two released reviewer figures.
- `example_data/`: tracked `(16, 256, 256)` raw, dense GT, sparse-positive and explicit-negative TIFFs.
- `model_adapters/`: the common CLI contract for all 12 models.
- `formal_runners/`: bundled Vanilla U-Net, MitoNet and official nnU-Net v2 bridge logic required by the adapters.
- `formal_assets/paired_roi_masks/`: the three fixed ROI masks, validity mask, logical hashes and installer used at runtime.
- `formal_assets/configs/mitonet/finetune_config_portable.yaml`: portable MitoNet runtime template.
- `formal_assets/sparseseg_adaptive_backend/`: optimized iterative-mask backend used by formal SparseSeg.
- `formal_assets/provenance/`: read-only source manifests, requested configs, historical result metadata, exact environment snapshots and hardware records.

## Formal design

Three fixed ROI-selection masks (budgets 1, 5 and 10) are paired with five training seeds (trial IDs 100--104), producing 15 cases. Trial IDs are seed repeats, not independent biological volumes. The canonical contract is `formal_assets/paired_roi_masks/fixed_paired_roi_masks.json`; an identical repository-side copy is `../fixed_paired_roi_masks.json`.

The installer expands the three compressed masks to the 15 exact filenames expected by the adapters. `run_formal_benchmark.py` verifies the raw, dense evaluation-only GT, explicit-negative mask, all input shapes, and every compression-independent logical mask hash before executing a model.

## Runtime files versus provenance

Formal execution reads only:

- `pipeline_config.example.json`;
- `model_adapters/` and `formal_runners/`;
- `formal_assets/paired_roi_masks/`;
- the portable MitoNet template;
- the SparseSeg adaptive backend;
- paths explicitly supplied through the command line or environment variables.

Everything under `formal_assets/provenance/` is read-only documentation of reported runs. Historical absolute paths may remain in those records, but neither runner opens or resolves them.

## Base environment

The local `requirements.txt` contains orchestration, evaluation and plotting requirements; model families still need separate environments.

```bash
conda create -n benchmark-pipeline python=3.10 -y
conda activate benchmark-pipeline
python -m pip install -r requirements.txt
```

Exact package snapshots are archived under `formal_assets/provenance/environments/`. Select CUDA-enabled PyTorch/TensorFlow builds compatible with the reader's NVIDIA driver.

### SparseSeg, SparseSeg-ViT and Vanilla U-Net

```bash
conda create -n vem-benchmark python=3.10 -y
conda activate vem-benchmark
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy scipy tifffile scikit-image scikit-learn zarr tensorboard einops transformers tqdm pyyaml pandas matplotlib
```

Set `VEM_PYTHON` to this environment's Python. The SparseSeg adapter locates `segment_cell.py` and `adaptive_iterated_mask.py` relative to the clone. Formal SparseSeg uses three 60-epoch iterations; SparseSeg-ViT uses five 50-epoch iterations. Vanilla U-Net raw uses its native sampler/loss. Vanilla U-Net sparse-matched changes only to SparseSeg's positive-centred sampler and sparse-aware loss and does not use SparseSeg features or iterative refinement.

### Official nnU-Net v2

```bash
conda create -n nnunetv2-benchmark python=3.10 -y
conda activate nnunetv2-benchmark
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install nnunetv2==2.8.1 tifffile scipy
```

Set `NNUNET_PYTHON`. `formal_runners/nnunetv2_official/` contains the tracked Tiff3DIO data bridge and sparse-matched trainer extension; no external or untracked code root is required. The planner, preprocessor, PlainConvUNet, optimizer, checkpoint and predictor remain official nnU-Net v2. Sparse-matched changes only the foreground-centred sampler and sparse-aware loss.

### MitoNet / Empanada

```bash
conda create -n mitonet-benchmark python=3.10 -y
conda activate mitonet-benchmark
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install empanada-napari==1.2.3 tifffile pyyaml
```

Download the official MitoNet_v1 checkpoint and base YAML. Set `MITONET_PYTHON`, `MITONET_MODEL` and `MITONET_BASE_CONFIG`. The adapter uses the tracked portable finetune template. MitoNet-pretrained performs inference only. MitoNet sparse-adapted starts from the official checkpoint and uses Empanada's native objective and sampling, not the SparseSeg sparse-matched controls.

### StarDist

```bash
conda create -n stardist-benchmark python=3.10 -y
conda activate stardist-benchmark
python -m pip install tensorflow stardist tifffile scipy
```

Set `STARDIST_PYTHON`. The adapter calls the official `StarDist3D` API; the formal schedule is 50 epochs x 100 steps.

### DeePiCt

```bash
git clone https://github.com/ZauggGroup/DeePiCt.git
conda create -n deepict-benchmark python=3.7 -y
conda activate deepict-benchmark
python -m pip install "snakemake==5.13.0" "keras==2.3.1" "tensorflow-gpu==2.0.0" mrcfile pyyaml tifffile numpy
```

Set `DEEPICT_PYTHON` and `DEEPICT_ROOT`. The adapter copies the official `2d_cnn/` workflow into its per-case work directory and materializes portable paths. A modern compatibility implementation is available only for the one-epoch interface check and is not the source of formal DeePiCt values.

### COSEM 2D/3D U-Net (CellMap)

```bash
conda create -n cellmap-benchmark -c conda-forge python=3.11 cxx-compiler -y
conda activate cellmap-benchmark
git clone https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git
python -m pip install -e cellmap-segmentation-challenge
python -m pip install tifffile
```

Set `CELLMAP_PYTHON`. The runtime adapter accepts only command-line inputs and a per-case work directory. Archived manifests and original run configs are retained under read-only provenance.

## Environment variables

Use environment-specific interpreter and external-checkpoint paths. Angle-bracket values are placeholders, not repository defaults.

```bash
export VEM_PYTHON=<vem-environment-python>
export NNUNET_PYTHON=<nnunet-environment-python>
export MITONET_PYTHON=<mitonet-environment-python>
export MITONET_MODEL=<MitoNet_v1-checkpoint>
export MITONET_BASE_CONFIG=<MitoNet_v1-yaml>
export STARDIST_PYTHON=<stardist-environment-python>
export DEEPICT_PYTHON=<deepict-environment-python>
export DEEPICT_ROOT=<DeePiCt-checkout>
export CELLMAP_PYTHON=<cellmap-environment-python>
```

The same variables can be defined in a PyCharm Run Configuration. Use this `benchmark_pipeline` folder as the working directory.

## Small executable interface check

```bash
python run_benchmark_pipeline.py --epochs 1
python run_benchmark_pipeline.py --epochs 1 --skip-existing
python run_benchmark_pipeline.py --epochs 1 --no-plots --models "Vanilla-UNet" "nnU-Net-SparseMatched"
```

Each adapter must write one binary prediction TIFF plus a `.timing.json` sidecar. The script evaluates the predictions into `outputs/example_evaluation_metrics.csv` and then redraws the released figures from `data/*.csv`.

## Formal 15-case reproduction

The full raw, dense evaluation-only GT and explicit-negative volume are not duplicated in Git. Place them under one directory with exact names:

```text
FORMAL_DATA_ROOT/
  hela2_em_s3.tif
  hela2_mito_s3.tif
  negative_hela2_em_s3.tif
```

All three must have shape `(200, 1500, 796)`. From this folder:

```bash
python formal_assets/paired_roi_masks/install_formal_masks.py --data-root "$FORMAL_DATA_ROOT"
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --validate-only
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --output-root formal_predictions --dry-run
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --output-root formal_predictions
```

`--models` and `--cases` select subsets. `--overwrite` is required to replace an existing prediction. Formal schedules, seeds and paired masks are fixed by the tracked runner and manifest.

Evaluate the completed formal layout:

```bash
python ../evaluation_cross_trials_extreme.py \
  --gt-path "$FORMAL_DATA_ROOT/hela2_mito_s3.tif" \
  --empanda-root formal_predictions \
  --output-dir formal_predictions/evaluation_cross_trials_extreme \
  --strict
```

With no `--model` option the evaluator selects exactly the formal 12-model set. Historical optional entries remain accessible only through explicit repeated `--model` arguments. The evaluator writes per-case absolute IoU, precision, recall, predicted foreground fraction, historical relative IoU, summaries, missing-file audit, and de-duplicated timing totals.

## Figure regeneration

The released figures can be recreated without prediction data:

```bash
python run_benchmark_pipeline.py --plots-only
```

Custom CSV paths are also supported:

```bash
python run_benchmark_pipeline.py --plots-only \
  --metrics-csv data/benchmark_metrics.csv \
  --time-csv data/time_per_epoch.csv \
  --wallclock-csv data/wallclock_total.csv \
  --figure-output-dir outputs
```

Equivalent direct plotting command:

```bash
python plot_benchmark_figures.py \
  --metrics data/benchmark_metrics.csv \
  --time data/time_per_epoch.csv \
  --wallclock data/wallclock_total.csv \
  --output-dir outputs
```

The plotting code validates table schemas and row counts before drawing:

- `Fig_add1_v3.pdf`: 10 models, relative IoU and time/epoch, excluding both sparse-matched controls and including the dashed MitoNet-pretrained median.
- `Fig_add_more1_v3.pdf`: all 12 models, relative IoU, absolute IoU, precision, recall, predicted foreground fraction and total end-to-end wall-clock.

## Timing provenance

Wall-clock values are end-to-end within the recorded run scope and exclude scheduler queue delay. The released measurements include multiple GPU types, so they describe computational cost and are not hardware-normalized speed rankings. Hardware and environment records are under `formal_assets/provenance/environments/`.
