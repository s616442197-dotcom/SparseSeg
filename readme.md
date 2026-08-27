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

The directory can be organized as:

```text
SparseSeg/
├── inputdata/
│   ├── raw.tif
│   └── sparse_positive_label.tif
├── benchmark/
│   ├── test_extreme.py
│   ├── test_ablation.py
│   ├── test_star.py
│   ├── train.sh
│   ├── run_deepict_finetune_batch.sh
│   ├── 2dtraining.py
│   └── 3dtraining.py
├── test_prediction.py
├── segment_cell.py
└── ...
```

Then run:

```bash
python test_prediction.py
```

This script provides a simple entry point for training and prediction using user-provided raw images and sparse positive annotations. If your file names or paths are different from the default settings, modify the corresponding file paths in `test_prediction.py`.

The output prediction masks and intermediate results will be saved to the output directory specified in `test_prediction.py`.

---

## Reproducing Paper Experiments

> **Reviewer 3 benchmark entry points.** The historical directory name
> `janelia_cosem/benchamark/` is retained for compatibility. Start with
> `README_EXTREME_BENCHMARKS.md` in that directory. It explains how to create
> isolated Python environments, install CUDA-enabled PyTorch and the common
> scientific packages, and install the separate MitoNet/Empanada and official
> nnU-Net v2 dependencies for the reported nnU-Net controls. No particular filesystem layout or job
> scheduler is required.
>
> Configure `--data-root` and `--output-root` for your machine and run the
> Python benchmark entry point once for each trial/ROI pair. The shell files in
> `benchamark/` record the experiment configurations used in the study, but
> users may invoke the underlying Python commands directly or adapt them to
> their own batch system. After predictions are placed in the documented output
> layout, run `python janelia_cosem/evaluation_cross_trials_extreme.py` with no
> arguments. It writes per-trial absolute IoU, precision, recall, predicted
> foreground fraction, historical log-normalized relative IoU, and
> de-duplicated end-to-end timing summaries. The relative-IoU definition is
> `ln(1 + 100 * IoU) / D`, where `D` is the maximum transformed IoU across all
> model/trial/ROI rows included in that evaluation run.
>
> The historical local `nnunet_2d` options in
> `sparse_baseline_benchmark.py` are retained only as legacy PlainConv2D audit
> code and are not reported as nnU-Net. The current benchmark uses the
> pip-installed official `nnunetv2==2.8.1` planner, preprocessor, trainer,
> checkpoint and predictor in `benchamark/nnunetv2_official/`.

In addition to the quick prediction example, we provide separate scripts for reproducing the main SparseSeg experiments and benchmark analyses reported in the paper.

The benchmark scripts used for the experiments reported in the paper are provided in the `benchmark/` folder. Each benchmark model may require its own software environment, because the compared methods depend on different packages and frameworks. Users should activate the corresponding environment before running each script.

Before running any benchmark script, users should check and modify the input paths, output paths, dataset names, and model-specific settings inside the corresponding script according to their local data organization.

---

### 1. Extreme sparse-annotation benchmark

The ROI-level sparse benchmark using 1, 5, and 10 positive ROIs can be reproduced using:

```bash
python janelia_cosem/test_extreme.py
```

This script evaluates SparseSeg under extreme sparse-annotation settings, corresponding to the benchmark in which only 1, 5, or 10 sparse positive ROIs are used as supervision.

Before running, users should check and modify the input paths inside `janelia_cosem/test_extreme.py`, for example:

```python
RAW_PATH = "path/to/raw_volume.tif"
MASK_PATH = "path/to/sparse_positive_label.tif"
OUTPUT_DIR = "path/to/output_folder"
```

Depending on the benchmark configuration, users may also modify the tested ROI numbers, random seeds, target structures, and number of repeats.

The script saves prediction masks, intermediate refinement results, and evaluation summaries to the configured output directory.

---

### 2. Ablation experiments

The ablation experiments can be reproduced using:

```bash
python janelia_cosem/test_ablation.py
```

This script reproduces the parameter and architecture ablation analyses reported in the paper, including tests for:

- shape-refinement threshold
- boundary / edge IoU filtering threshold
- low-weight coefficient
- kernel-size configuration
- number of kernels

Before running, users should check and modify the input paths inside `janelia_cosem/test_ablation.py`, for example:

```python
RAW_PATH = "path/to/raw_volume.tif"
MASK_PATH = "path/to/sparse_positive_label.tif"
OUTPUT_DIR = "path/to/output_folder"
```

The output files include prediction masks, intermediate refinement results, and quantitative evaluation tables for different parameter settings.

---

### 3. Full iterative SparseSeg experiment

For full iterative training and prediction on a user-defined dataset, run:

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

This command performs iterative training, prediction, geometry-based refinement, and pseudo-label expansion.

---

### 4. Benchmark scripts for paper results

The paper compares SparseSeg with several external segmentation frameworks, including MitoNet, nnU-Net, COSEM / CellMap 2D and 3D U-Net, StarDist, DeePiCt, and SparseSeg-ViT.

The main benchmark scripts include:

| Benchmark | Script | Description |
| --------- | ------ | ----------- |
| SparseSeg extreme sparse benchmark | `janelia_cosem/test_extreme.py` | Reproduces the 1, 5, and 10 ROI sparse-annotation benchmark for SparseSeg |
| SparseSeg ablation experiments | `janelia_cosem/test_ablation.py` | Reproduces the ablation experiments for shape threshold, edge IoU threshold, low-weight coefficient, and kernel configuration |
| StarDist benchmark | `janelia_cosem/benchamark/test_star.py` | Trains and evaluates StarDist3D on the same sparse-label benchmark settings |
| DeePiCt benchmark | `janelia_cosem/benchamark/run_deepict_finetune_batch.sh` | Runs DeePiCt fine-tuning and prediction for the sparse-label benchmark |
| COSEM / CellMap 2D U-Net benchmark | `janelia_cosem/benchamark/2dtraining.py` | Trains and predicts using a 2D U-Net baseline |
| COSEM / CellMap 3D U-Net benchmark | `janelia_cosem/benchamark/3dtraining.py` | Trains a 3D U-Net baseline |
| nnU-Net v2 raw and sparse-matched controls | `janelia_cosem/benchamark/nnunetv2_official/submit_official_nnunetv2_pipeline.sh` | Runs official nnU-Net v2 data conversion, planning, preprocessing, training, checkpointing and whole-volume prediction |

These scripts are intended to reproduce the benchmark settings reported in the paper when the same input volumes, sparse labels, and random seeds are used.

---

### 5. Running third-party benchmark models

For third-party methods, users should install the corresponding official software environments before running the benchmark scripts.

Recommended environments include:

| Method | Recommended environment |
| ------ | ----------------------- |
| StarDist | StarDist / TensorFlow environment |
| DeePiCt | DeePiCt official environment |
| nnU-Net | nnU-Net v2 environment |
| COSEM / CellMap 2D and 3D U-Net | CellMap segmentation challenge environment |
| MitoNet | Official empanada-napari environment |

For StarDist, DeePiCt, nnU-Net, and COSEM / CellMap U-Net baselines, the corresponding benchmark scripts are provided in the `benchmark/` folder. Users should activate the required environment and then run the corresponding script.

For example:

```bash
# StarDist benchmark
conda activate stardist_env
python janelia_cosem/benchamark/test_star.py

# DeePiCt benchmark
conda activate deepict_env
bash janelia_cosem/benchamark/run_deepict_finetune_batch.sh

# official nnU-Net v2 benchmark (Slurm example)
python -m venv nnunetv2_official
nnunetv2_official/bin/python -m pip install nnunetv2==2.8.1
bash janelia_cosem/benchamark/nnunetv2_official/submit_official_nnunetv2_pipeline.sh

# COSEM / CellMap 2D U-Net benchmark
conda activate cellmap_env
python janelia_cosem/benchamark/2dtraining.py

# COSEM / CellMap 3D U-Net benchmark
conda activate cellmap_env
python janelia_cosem/benchamark/3dtraining.py
```

MitoNet was evaluated using the official empanada-napari implementation. To reproduce the MitoNet benchmark, users should install the official empanada-napari plugin, load the pretrained `MitoNet_v1` model, and run inference using the parameter settings reported in the Appendix of the paper. The resulting prediction TIFF files can then be evaluated using the same IoU-based evaluation scripts used for the other benchmark models.

Because different benchmark models use different training pipelines and input formats, some scripts perform format conversion automatically. For example, the DeePiCt benchmark converts TIFF volumes and labels into MRC format before training and converts the predicted MRC files back to TIFF for evaluation. The official nnU-Net v2 entry point creates its Tiff3DIO datasets, runs fingerprint extraction and planning, and writes evaluator-compatible binary TIFF predictions automatically.

All benchmark outputs should be saved as TIFF prediction masks or converted into TIFF format before quantitative evaluation. The same evaluation scripts can then be used to compute IoU, normalized relative IoU, and other summary statistics reported in the paper.

---

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

Approximate runtime depends on the volume size, patch scale, number of training iterations, and number of benchmark settings. As a rough guide:

| Experiment | Script | Approximate runtime |
| ---------- | ------ | ------------------- |
| Quick prediction on one small volume | `test_prediction.py` | tens of minutes |
| Full iterative SparseSeg run | `iterative_bash.py` | tens of minutes to a few hours per dataset |
| Extreme sparse-annotation benchmark | `janelia_cosem/test_extreme.py` | several hours depending on ROI settings and repeats |
| Ablation experiments | `janelia_cosem/test_ablation.py` | several hours to overnight depending on the number of tested parameters |
| StarDist benchmark | `janelia_cosem/benchamark/test_star.py` | several hours depending on volume size and number of repeats |
| DeePiCt benchmark | `janelia_cosem/benchamark/run_deepict_finetune_batch.sh` | several hours to overnight depending on the number of ROI settings |
| nnU-Net benchmark | `janelia_cosem/benchamark/train.sh` | several hours to more than one day depending on dataset size |
| COSEM / CellMap U-Net benchmarks | `janelia_cosem/benchamark/2dtraining.py`, `janelia_cosem/benchamark/3dtraining.py` | several hours to overnight depending on 2D or 3D configuration |

Benchmark scripts in the `benchmark/` folder may have different runtime and memory requirements depending on the corresponding model. In general, SparseSeg, StarDist, DeePiCt, nnU-Net, and COSEM / CellMap U-Net baselines should be run in their own recommended environments. MitoNet inference was performed through the official empanada-napari interface using the parameter settings listed in the Appendix.

For large volumes or limited GPU memory, users can reduce `patch_scale`, reduce the number of iterations, or process smaller sub-volumes.

---

## Notes

- The raw TIFF and sparse positive-label TIFF should have matching spatial dimensions.
- Sparse positive labels should mark only representative target regions.
- Unlabeled regions are treated as unknown rather than dense background.
- For large volumes, GPU acceleration is recommended.
- If memory is limited, reduce `patch_scale` or process the volume in smaller regions.
- For reproducible benchmarking, use fixed random seeds when selecting sparse ROIs.
- Benchmark scripts for paper results are provided in the `benchmark/` folder and should be run in the corresponding software environments.
- MitoNet results should be reproduced using the official empanada-napari implementation and the Appendix parameter settings.

---

[//]: # "## Citation"
[//]: #
[//]: # "```bibtex"
[//]: # "Coming soon"
[//]: # "```"
