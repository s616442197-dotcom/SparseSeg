<div align="center">

# SparseSeg

### Target-Conditioned Sparse Annotation Segmentation for Cryo-Volume Electron Microscopy

![Python](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Application](https://img.shields.io/badge/Application-Cryo--vEM-blue.svg?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Segmentation-green.svg?style=for-the-badge)

SparseSeg is a target-conditioned, sparsity-driven segmentation framework for **Cryo-Volume Electron Microscopy (cryo-vEM)**.

[Overview](#overview) • [Quick Start](#quick-start) • [Parameters](#parameters)

---

</div>

## Overview

Cryo-volume electron microscopy (cryo-vEM) enables near-native visualization of cellular ultrastructure. However, its broad application is severely limited by the low contrast and prohibitive cost of dense voxel-level annotation.

Most existing automated segmentation methods, developed or trained on conventional vEM datasets, are difficult to generalize across different cell types and imaging conditions.

**SparseSeg** is a target-conditioned, sparsity-driven segmentation framework that reconceptualizes organelle segmentation as a **discovery process** rather than a closed-set classification task. Instead of relying on dense annotations, SparseSeg uses a small number of context-specific exemplars to iteratively propagate reliable supervision throughout the volume.

The framework integrates sparse patch-based sampling, a multi-kernel U-Net architecture optimized for cryogenic preserved images, and geometry-consistent refinement to progressively expand accurate segmentation while suppressing context-dependent false positives.

SparseSeg is evaluated on serial cryo-FIB-SEM datasets spanning multiple cell types, organelles, and annotation sparsity regimes, including extreme few-shot settings with **less than 1% labeled slices**.

---

## Quick Start

### Command line

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

### Python script

```python
from segment_cell import main

for inter_idx in range(5):
    print(f"\n=== Running iteration {inter_idx} ===")

    main(
        interation_idx=inter_idx,
        z_threshold=1,
        patch_scale=80,
        raw_name="your_raw_name",
        mask_name="your_mask_name",
        folder_name="your_folder_name",
        area_coef=1.0,
        edge_coef=1.0,
        iou_thresh=0.6,
        threshold=0.9,
        negative_threshold=3,
        low_weight_coeff=200,
        sparsity_weight=0.5,
        filer_method=2
    )
```

---

## Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `raw_name` | Raw cryo-vEM volume name | Required |
| `mask_name` | Sparse annotation mask name | Required |
| `folder_name` | Output folder name | Required |
| `interation_idx` | int | Current iterative refinement round | **Required** |
| `filer_method` | int | Filtering / preprocessing strategy | `2` |
| `z_threshold` | int | Slice-level confidence threshold | `10` |
| `patch_scale` | int | Patch size used for training and inference | `140` |
| `area_coef` | float | Area consistency coefficient | `1.0` |
| `edge_coef` | float | Edge consistency coefficient | `0.5` |
| `iou_thresh` | float | IoU threshold for pseudo-label refinement | `0.6` |
| `threshold` | float | Prediction confidence threshold | `0.5` |
| `negative_threshold` | float | Threshold for suppressing false positives | `1.0` |
| `low_weight_coeff` | float | Weight for low-confidence regions | `10.0` |
| `sparsity_weight` | float | Sparsity regularization weight | `0.0` |

---

[//]: # (## Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # (Coming soon)

[//]: # (```)