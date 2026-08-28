<div align="center">

# SparseSeg

### Target-Conditioned Sparse Annotation Segmentation for Cryo-Volume Electron Microscopy

![Python](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Application](https://img.shields.io/badge/Application-Cryo--vEM-blue.svg?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Segmentation-green.svg?style=for-the-badge)

SparseSeg is a target-conditioned, sparsity-driven segmentation framework for **Cryo-Volume Electron Microscopy (cryo-vEM)**.

[Overview](#overview) • [Installation](#installation) • [Quick Start](#quick-start) • [Reproducing Paper Experiments](#reproducing-paper-experiments) • [Input Data Format](#input-data-format) • [Parameters](#parameters) • [Compute Requirements](#compute-requirements)

---

</div>

## Overview

Cryo-volume electron microscopy (cryo-vEM) enables near-native visualization of cellular ultrastructure. However, its broad application is severely limited by the low contrast of cryogenic preserved images and the prohibitive cost of dense voxel-level annotation.

Most existing automated segmentation methods, developed or trained on conventional vEM datasets, are difficult to generalize across different cell types, organelles, and imaging conditions.

**SparseSeg** is a target-conditioned, sparsity-driven segmentation framework that reconceptualizes organelle segmentation as a **discovery process** rather than a closed-set classification task. Instead of relying on dense annotations, SparseSeg uses a small number of context-specific positive exemplars to iteratively propagate reliable supervision throughout the volume.

The framework integrates sparse patch-based sampling, a multi-kernel U-Net architecture optimized for cryogenic preserved images, and geometry-consistent refinement to progressively expand accurate segmentation while suppressing context-dependent false positives.

SparseSeg is evaluated on serial cryo-FIB-SEM datasets spanning multiple cell types, organelles, and annotation sparsity regimes, including extreme few-shot settings with **less than 1% labeled slices**.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/s616442197-dotcom/SparseSeg.git
cd SparseSeg
```

### 2. Create a conda environment

We recommend using Python 3.10.

```bash
conda create -n sparseseg python=3.10 -y
conda activate sparseseg
```

### 3. Install PyTorch

Install the PyTorch version compatible with your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only installation, use:

```bash
pip install torch torchvision torchaudio
```

### 4. Install required Python packages

```bash
pip install numpy scipy pandas scikit-image tifffile matplotlib tqdm opencv-python scikit-learn
```

If a `requirements.txt` file is provided, the dependencies can also be installed by:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Quick prediction with your own data

For a quick test, place your raw cryo-vEM TIFF file and the corresponding sparse positive-label TIFF file into the `inputdata` directory.

The minimal user-data layout is:

```text
SparseSeg/
├── inputdata/
│   ├── raw.tif
│   └── sparse_positive_label.tif
├── janelia_cosem/
│   ├── segment_cell.py
│   ├── adaptive_iterated_mask.py
│   ├── test_prediction.py
│   └── benchmark_pipeline/
└── readme.md
```
Then run:

```bash
python test_prediction.py
```

This script provides a simple entry point for training and prediction using user-provided raw images and sparse positive annotations. If your file names or paths are different from the default settings, modify the corresponding file paths in `test_prediction.py`.

The output prediction masks and intermediate results will be saved to the output directory specified in `test_prediction.py`.

---

## Reproducing Paper Experiments

The maintained Reviewer 3 interface is `janelia_cosem/benchmark_pipeline/`. All benchmark reproduction commands below use the tracked formal runner, evaluator, and CSV plotting workflow.

The formal package contains:

- `run_formal_benchmark.py`: expands the fixed 12-model x 15-case grid and writes evaluator-compatible predictions;
- `run_benchmark_pipeline.py`: runs the small one-epoch interface example or redraws figures from released CSV tables;
- `README.md`: model-specific installation, environment, input, timing, and output instructions;
- `formal_assets/paired_roi_masks/fixed_paired_roi_masks.json`: the paired ROI masks, seeds, and logical hashes used by the formal runs.

Install the orchestration and plotting dependencies, then enter the maintained folder:

```bash
python -m pip install -r janelia_cosem/benchmark_pipeline/requirements.txt
cd janelia_cosem/benchmark_pipeline
```

Redraw both released reviewer figures without model environments or raw data:

```bash
python run_benchmark_pipeline.py --plots-only
```

After installing the model-specific environments described in `benchmark_pipeline/README.md`, run the packaged one-epoch interface check:

```bash
python run_benchmark_pipeline.py --epochs 1
```

For the formal experiment, first download and validate the public `jrc_hela-2` raw EM and dense mitochondria GT ([DOI: 10.25378/janelia.13114211](https://doi.org/10.25378/janelia.13114211)):

```bash
python prepare_formal_inputs.py --data-root "$FORMAL_DATA_ROOT"
```

The downloader reads `em/fibsem-uint16/s3` and `labels/mito_seg/s3` from the public Janelia COSEM N5 store, transposes them to `(Z,H,W)=(200,1500,796)`, and checks the archived compression-independent logical SHA-256 values. The exact hashes and public paths are documented in `benchmark_pipeline/README.md` and `fixed_paired_roi_masks.json`.

The explicit-negative input is author-generated rather than downloaded: open the same raw stack in Fiji/ImageJ, run `../preprocessing.ijm`, draw the requested negative ROIs, and use the saved `StackC.tif` as `negative_hela2_em_s3.tif`. For exact reproduction of the paper, use the author-generated negative mask supplied with Source Data; a newly drawn mask defines a new experiment. The staging command is:

```bash
python prepare_formal_inputs.py \
  --data-root "$FORMAL_DATA_ROOT" \
  --negative-from /path/to/StackC.tif \
  --require-negative
```

Then install the released fixed ROI masks, validate all inputs, and expand or execute the complete command grid:

```bash
python prepare_formal_inputs.py --data-root "$FORMAL_DATA_ROOT" --validate-only --require-negative
python formal_assets/paired_roi_masks/install_formal_masks.py --data-root "$FORMAL_DATA_ROOT"
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --validate-only
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --output-root formal_predictions --dry-run
python run_formal_benchmark.py --data-root "$FORMAL_DATA_ROOT" --output-root formal_predictions
```

Evaluate the formal prediction layout with the tracked evaluator:

```bash
python ../evaluation_cross_trials_extreme.py \
  --gt-path "$FORMAL_DATA_ROOT/hela2_mito_s3.tif" \
  --empanda-root formal_predictions \
  --output-dir formal_predictions/evaluation_cross_trials_extreme \
  --strict
```

The evaluator reports every trial's absolute IoU, precision, recall, predicted foreground fraction, historical log-normalized relative IoU, and available end-to-end wall-clock timing. See `janelia_cosem/benchmark_pipeline/README.md` for the exact 12 model names, separate environments, formal schedules, and optional subset commands.

## Input Data Format

SparseSeg expects two TIFF files as input:

| File | Description |
| ---- | ----------- |
| Raw TIFF | Raw cryo-vEM volume |
| Sparse positive-label TIFF | Sparse annotation mask containing user-provided positive labels |

The recommended input shape is:

```text
(Z, H, W)
```

where `Z` is the number of slices and `H, W` are the image height and width.

The sparse positive-label TIFF should have the same spatial size as the raw TIFF. Non-zero pixels are treated as positive annotations of the target structure. Unlabeled pixels are treated as unknown regions rather than dense negative labels.

For multi-channel TIFF files, users may need to modify `test_prediction.py` to select the desired channel before training and prediction.

---

## Command Line Usage

For full iterative refinement, run:

```bash
cd janelia_cosem

python iterative_bash.py \
    --raw_name "${your_raw_name}" \
    --mask_name "${your_mask_name}" \
    --folder_name "${your_folder_name}" \
    --patch_scale 80 \
    --sparsity_weight 0.5 \
    --z_threshold 10 \
    --iou_thresh 0.6 \
    --threshold 0.9 \
    --area_coef 1.0 \
    --edge_coef 1.0 \
    --negative_threshold 3 \
    --low_weight_coeff 200 \
    --num_iterations 5
```

---

## Python Script Usage

```python
from segment_cell import main

for inter_idx in range(5):
    print(f"\n=== Running iteration {inter_idx} ===")

    main(
        interation_idx=inter_idx,
        z_threshold=1,
        patch_scale=140,
        raw_name="your_raw.tif",
        mask_name="your_positive_mask.tif",
        folder_name="folder_to_store",
        area_coef=1.0,
        edge_coef=1.0,
        iou_thresh=0.6,
        threshold=0.01,
        negative_threshold=3,
        low_weight_coeff=50,
        sparsity_weight=1.0,
    )
```

---

## Parameters

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `raw_name` | Raw cryo-vEM volume name | Required |
| `mask_name` | Sparse positive-label mask name | Required |
| `folder_name` | Output folder name | Required |
| `interation_idx` | Current iterative refinement round | Required |
| `filer_method` | Filtering / preprocessing strategy | `2` |
| `z_threshold` | Slice-level confidence threshold | `10` |
| `patch_scale` | Patch size used for training and inference | `80` |
| `area_coef` | Area consistency coefficient | `1.0` |
| `edge_coef` | Edge consistency coefficient | `1.0` |
| `iou_thresh` | IoU threshold for pseudo-label refinement | `0.6` |
| `threshold` | Shape or prediction filtering threshold | `0.05` |
| `negative_threshold` | Threshold for suppressing false positives | `3.0` |
| `low_weight_coeff` | Weight for low-confidence or distant regions | `50.0` |
| `sparsity_weight` | Sparsity regularization weight | `1.0` |

---

## Expected Output

After running `test_prediction.py` or the iterative pipeline, SparseSeg will generate prediction masks and intermediate refinement results in the specified output folder.

The exact output file names may depend on the settings in the script, but typically include:

```text
output_folder/
├── volume_mask_pred.tiff
├── newbase_masks.tiff
├── model_0.pt
├── model_1.pt
└── ...
```

For benchmark and ablation scripts, the output folder may additionally contain quantitative summary tables, parameter-specific results, and visualization files.

---

## Compute Requirements

The experiments reported in the paper were performed on a workstation with the following configuration:

| Component | Configuration |
| --------- | ------------- |
| CPU | AMD Ryzen Threadripper 7970X, 32 cores, 4.00 GHz |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 5080 |
| GPU memory | 16 GB VRAM |
| Storage | 5.46 TB local storage |

A CUDA-enabled GPU is recommended for reproducing the benchmark experiments. In practice, a GPU with at least 12--16 GB VRAM is recommended for training and inference on large vEM volumes.

Approximate runtime depends on volume size, patch sampling, model family, and iteration count:

| Experiment | Maintained entry point | Typical scale |
| ---------- | ---------------------- | ------------- |
| Quick prediction on one small volume | `janelia_cosem/test_prediction.py` | tens of minutes |
| Full iterative SparseSeg run | `janelia_cosem/iterative_bash.py` | tens of minutes to a few hours per dataset |
| One-epoch 12-model interface check | `janelia_cosem/benchmark_pipeline/run_benchmark_pipeline.py` | model- and environment-dependent |
| Formal 12-model x 15-case benchmark | `janelia_cosem/benchmark_pipeline/run_formal_benchmark.py` | hours to days; use a suitable GPU batch system if available |
| Figure-only reproduction | `janelia_cosem/benchmark_pipeline/run_benchmark_pipeline.py --plots-only` | seconds to minutes; no GPU required |

The released timing tables contain the measured hardware provenance. Because model families were run on different GPU types, wall-clock values are descriptive computational-cost measurements rather than hardware-normalized speed rankings.
For large volumes or limited GPU memory, users can reduce `patch_scale`, reduce the number of iterations, or process smaller sub-volumes.

---

## Notes

- The raw TIFF and sparse positive-label TIFF should have matching spatial dimensions.
- Sparse positive labels should mark only representative target regions.
- Unlabeled regions are treated as unknown rather than dense background.
- For large volumes, GPU acceleration is recommended.
- If memory is limited, reduce `patch_scale` or process the volume in smaller regions.
- For reproducible benchmarking, use fixed random seeds when selecting sparse ROIs.
- Paper benchmark reproduction uses `janelia_cosem/benchmark_pipeline/`; install each model family in the separate environment documented there.
- MitoNet results should be reproduced using the official empanada-napari implementation and the Appendix parameter settings.

---

[//]: # "## Citation"
[//]: #
[//]: # "```bibtex"
[//]: # "Coming soon"
[//]: # "```"
