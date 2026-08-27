# 12-model benchmark reproduction pipeline

This folder contains two separate workflows:

1. `run_benchmark_pipeline.py --epochs 1` is a small executable check on the packaged EM subvolume. It runs the 12 real model adapters, writes example predictions and metrics, then redraws the figures. These one-epoch values are not manuscript results.
2. `run_formal_benchmark.py` reproduces the formal 12-model by 15-case grid from the archived ROI masks, seeds, model configurations and environment records. Its output layout is directly readable by `vemmodel/janelia_cosem/evaluation_cross_trials_extreme.py`.

The plotting script reads only the released tables in `data/`; it never substitutes example-run values for formal values.

## Contents

- `data/benchmark_metrics.csv`: 12 models x 5 training seeds x 3 ROI budgets (180 metric rows).
- `data/time_per_epoch.csv`: released per-epoch timings used by `Fig_add1_v3.pdf`.
- `data/wallclock_total.csv`: total end-to-end wall-clock values used by `Fig_add_more1_v3.pdf`.
- `plot_benchmark_figures.py`: standalone generator for both reviewer figures.
- `outputs/Fig_add1_v3.pdf`: 10-model relative-IoU and time/epoch figure. Panel a includes the dashed MitoNet-pretrained median reference.
- `outputs/Fig_add_more1_v3.pdf`: 12-model relative IoU, absolute IoU, precision, recall, predicted foreground fraction and end-to-end wall-clock figure.
- `example_data/`: real `(16, 256, 256)` raw, dense GT, sparse positive label and explicit negative label for the one-epoch interface check.
- `model_adapters/`: unified adapters for all 12 entries.
- `formal_assets/paired_roi_masks/`: fixed ROI masks, trial/seed manifest, hashes and installer.
- `formal_assets/configs/`: archived nnU-Net `dataset.json`/manifests, CellMap manifests/validity masks and run configs, MitoNet finetune configs, and complete DeePiCt configs.
- `formal_assets/sparseseg_adaptive_backend/`: archived optimized iterative-mask backend used by the formal SparseSeg run.
- `formal_assets/environments/`: exact `pip freeze` snapshots and hardware provenance.

## Statistical design of the 15 cases

The packaged controlled-replay grid contains three fixed ROI-selection masks (ROI budgets 1, 5 and 10) paired with five training seeds (trial IDs 100--104), giving 15 cases. The trial IDs are seed repeats; they are not five independent biological volumes or five additional ROI masks. `formal_assets/paired_roi_masks/fixed_paired_roi_masks.json` is the canonical machine-readable definition. The same JSON is also present at `vemmodel/janelia_cosem/fixed_paired_roi_masks.json` for repository-side runners.

The installer expands the three archived TIFF assets to the 15 exact filenames expected by the adapters without changing their pixels. SHA-256 hashes, shapes and seeds are verified.

The released formal runner validates the full-volume input shape and compression-independent logical hashes for all 15 installed sparse masks before any model starts. The released CellMap 2D/3D metric and timing rows were regenerated with this canonical fixed-mask grid.

## Environments

The root `requirements.txt` intentionally contains only minimum version ranges for plotting and orchestration. It is not a lockfile and is not sufficient for every model:

```bash
conda create -n benchmark-pipeline python=3.10 -y
conda activate benchmark-pipeline
python -m pip install -r requirements.txt
```

Each model family must use a separate environment. Exact combinations used for the reported runs are archived under `formal_assets/environments/`; this includes the PyTorch/CUDA, TensorFlow, official nnU-Net v2, Empanada/MitoNet, StarDist, DeePiCt and CellMap environments. Recreate the recorded environment first, then select a CUDA build compatible with the target machine's NVIDIA driver.

### VEM / SparseSeg / SparseSeg-ViT / Vanilla U-Net

```bash
conda create -n vem-benchmark python=3.10 -y
conda activate vem-benchmark
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy scipy tifffile scikit-image scikit-learn zarr tensorboard einops transformers tqdm pyyaml pandas matplotlib
```

Set `VEMMODEL_ROOT` and `VEM_PYTHON`. Formal SparseSeg CNN uses three iterations of 60 epochs and the archived iterative-mask backend; SparseSeg-ViT uses five iterations of 50 epochs. Vanilla U-Net raw uses its native sampler/loss. Vanilla U-Net sparse-matched keeps the vanilla architecture but uses the SparseSeg positive-centered sampler and sparse-aware loss without SparseSeg's extra features or iterative refinement.

### Official nnU-Net v2

```bash
conda create -n nnunetv2-benchmark python=3.10 -y
conda activate nnunetv2-benchmark
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install nnunetv2==2.8.1 tifffile
```

Set `NNUNET_PYTHON` and `NNUNET_CODE_ROOT`. The code root must contain `prepare_official_nnunetv2_data.py` and `nnunet_ext_trainers/`. Raw and sparse-matched variants use the archived dataset IDs and metadata under `formal_assets/configs/nnunet/`. Sparse-matched changes only sampler/loss and does not use SparseSeg features or iterative refinement.

### MitoNet / Empanada

```bash
conda create -n mitonet-benchmark python=3.10 -y
conda activate mitonet-benchmark
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install empanada-napari==1.2.3 tifffile pyyaml
```

Download the official MitoNet_v1 checkpoint and base YAML; set `MITONET_PYTHON`, `MITONET_MODEL`, `MITONET_BASE_CONFIG` and `MITONET_FINETUNE_CONFIG`. All 15 requested finetune configs are archived under `formal_assets/configs/mitonet/`; their original path fields are retained as provenance, while the adapter replaces train, model-output and checkpoint paths from its command-line arguments. MitoNet-pretrained performs inference only. MitoNet sparse-adapted starts a fresh task-specific optimization run initialized from the official checkpoint and uses the native Empanada objective/sampling, not the SparseSeg sparse-aware loss/sampler.

### StarDist

```bash
conda create -n stardist-benchmark python=3.10 -y
conda activate stardist-benchmark
python -m pip install --upgrade pip
python -m pip install tensorflow stardist tifffile scipy
```

Set `STARDIST_PYTHON`. The adapter calls the official `StarDist3D` API. The formal schedule is 50 epochs and 100 steps per epoch. The exact legacy package set is in `formal_assets/environments/stardist_pip_freeze.txt`.

### DeePiCt

```bash
git clone https://github.com/ZauggGroup/DeePiCt.git
conda create -n deepict-benchmark python=3.7 -y
conda activate deepict-benchmark
python -m pip install "snakemake==5.13.0" "keras==2.3.1" "tensorflow-gpu==2.0.0" mrcfile pyyaml tifffile numpy
```

Set `DEEPICT_PYTHON` and `DEEPICT_ROOT`. Formal configs/data tables for all 15 cases are in `formal_assets/configs/deepict/`. The adapter copies `2d_cnn/` into its work directory before invoking `deploy_local.sh`. A modern compatibility path exists only for the one-epoch interface check and must not be treated as the formal DeePiCt benchmark.

### COSEM 2D/3D U-Net (CellMap)

```bash
conda create -n cellmap-benchmark -c conda-forge python=3.11 cxx-compiler -y
conda activate cellmap-benchmark
git clone https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git
python -m pip install -e cellmap-segmentation-challenge
python -m pip install tifffile
```

Set `CELLMAP_PYTHON`. Formal manifests, 2D train/predict configs and validity mapping are under `formal_assets/configs/cellmap/`. The archived all-voxel validity mask records the native legacy baseline treatment of unlabeled voxels as background; it is not a sparse-aware loss.

## Environment variables

```bash
export VEMMODEL_ROOT=/absolute/path/to/VEMModel
export VEM_PYTHON=/absolute/path/to/envs/vem-benchmark/bin/python
export NNUNET_PYTHON=/absolute/path/to/envs/nnunetv2-benchmark/bin/python
export NNUNET_CODE_ROOT=$VEMMODEL_ROOT/vemmodel/janelia_cosem/benchamark/nnunetv2_official/code
export MITONET_PYTHON=/absolute/path/to/envs/mitonet-benchmark/bin/python
export MITONET_MODEL=/absolute/path/to/MitoNet_v1.pth
export MITONET_BASE_CONFIG=/absolute/path/to/MitoNet_v1.yaml
export MITONET_FINETUNE_CONFIG=/absolute/path/to/benchmark_pipeline/formal_assets/configs/mitonet/finetune_config_portable.yaml
export STARDIST_PYTHON=/absolute/path/to/envs/stardist-benchmark/bin/python
export DEEPICT_PYTHON=/absolute/path/to/envs/deepict-benchmark/bin/python
export DEEPICT_ROOT=/absolute/path/to/DeePiCt
export CELLMAP_PYTHON=/absolute/path/to/envs/cellmap-benchmark/bin/python
```

The same variables can be placed in a PyCharm Run Configuration. Use this folder as the working directory.

## One-epoch executable check

```bash
python run_benchmark_pipeline.py --plots-only
python run_benchmark_pipeline.py --epochs 1
python run_benchmark_pipeline.py --epochs 1 --skip-existing
python run_benchmark_pipeline.py --epochs 1 --no-plots --models "Vanilla-UNet" "nnU-Net-SparseMatched"
```

## Formal 15-case reproduction

The large full-volume raw, dense evaluation GT and explicit-negative volume are not duplicated in this plotting package. Put them in one directory with these exact names:

```text
FORMAL_DATA_ROOT/
  hela2_em_s3.tif
  hela2_mito_s3.tif
  negative_hela2_em_s3.tif
```

All three must have shape `(200, 1500, 796)`. Install and validate the archived formal masks:

```bash
python formal_assets/paired_roi_masks/install_formal_masks.py --data-root "$FORMAL_DATA_ROOT"
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --validate-only
```

Run all 12 models over all 15 cases:

```bash
python run_formal_benchmark.py \
  --data-root "$FORMAL_DATA_ROOT" \
  --output-root /absolute/path/to/formal_predictions
```

`--models` and `--cases` select subsets; `--dry-run` prints commands; `--overwrite` is required to replace an existing prediction. Formal schedules, seeds and paired masks come from the archived manifest/configs rather than the one-epoch example settings.

Evaluate the generated, evaluator-compatible layout:

```bash
python "$VEMMODEL_ROOT/vemmodel/janelia_cosem/evaluation_cross_trials_extreme.py" \
  --gt-path "$FORMAL_DATA_ROOT/hela2_mito_s3.tif" \
  --empanda-root /absolute/path/to/formal_predictions \
  --output-dir /absolute/path/to/formal_predictions/evaluation_cross_trials_extreme \
  --strict
```

The evaluator writes per-case absolute IoU, precision, recall, predicted foreground fraction and historical relative IoU. The formal runner additionally records end-to-end wall-clock seconds per model/case.

## Timing and hardware provenance

Timing values are end-to-end within their recorded run scope and exclude scheduler queue delay. SparseSeg finetuning reports the complete three-iteration trial-100/ROI-1 pipeline (`3453.270622` s). Legacy StarDist and DeePiCt controls ran on an RTX 5080 workstation; the fresh fixed-mask CellMap replay and the new controls used RTX 4090 resources; fresh formal SparseSeg cases used RTX 4090 and RTX A6000 resources. Because hardware is mixed, wall-clock values are descriptive computational-cost measurements and are not hardware-normalized speed rankings. See `formal_assets/environments/hardware_provenance.csv`.

## Figure regeneration

```bash
python plot_benchmark_figures.py
```

### CSV-only figure generation

When prediction and evaluation have already produced the three tables, no model environment or raw volume is needed for plotting:

```bash
python run_benchmark_pipeline.py --plots-only \
  --metrics-csv /absolute/path/to/benchmark_metrics.csv \
  --time-csv /absolute/path/to/time_per_epoch.csv \
  --wallclock-csv /absolute/path/to/wallclock_total.csv \
  --figure-output-dir /absolute/path/to/figures
```

The equivalent direct plotting entry point is:

```bash
python plot_benchmark_figures.py \
  --metrics /absolute/path/to/benchmark_metrics.csv \
  --time /absolute/path/to/time_per_epoch.csv \
  --wallclock /absolute/path/to/wallclock_total.csv \
  --output-dir /absolute/path/to/figures
```


The script validates all three source tables before drawing:

- `Fig_add1_v3.pdf`: 10 models, relative IoU and time/epoch; excludes the two sparse-matched controls and shows the MitoNet-pretrained median as a dashed line.
- `Fig_add_more1_v3.pdf`: all 12 models, relative IoU, absolute IoU, precision, recall, predicted foreground fraction and total end-to-end wall-clock.
