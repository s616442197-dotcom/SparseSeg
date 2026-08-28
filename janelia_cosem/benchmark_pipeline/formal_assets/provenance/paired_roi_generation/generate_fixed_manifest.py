#!/usr/bin/env python3
"""Generate the portable fixed paired-ROI/seed manifest used by the pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


HERE = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "source_paired_roi_manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "fixed_paired_roi_masks.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    masks: dict[str, dict[str, object]] = {}
    for roi in (1, 5, 10):
        path = HERE / "masks" / f"paired_roi_{roi}.tif"
        array = np.asarray(tifffile.imread(path), dtype=np.uint8)
        masks[str(roi)] = {
            "asset": f"masks/paired_roi_{roi}.tif",
            "shape_zyx": list(array.shape),
            "foreground_voxels": int(np.count_nonzero(array)),
            "logical_uint8_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "compressed_tiff_sha256": file_sha256(path),
        }

    rows: list[dict[str, object]] = []
    for item in source["rows"]:
        trial, roi = int(item["trial"]), int(item["roi_num"])
        base_seed = trial * 100 + roi
        rows.append(
            {
                "case_id": f"{trial}_{roi}",
                "trial": trial,
                "roi_num": roi,
                "mask_asset": masks[str(roi)]["asset"],
                "installed_sparse_filename": f"label_hela2_mito_{trial}_{roi}.tif",
                "source_uncompressed_tiff_sha256": item["sparse_mask_sha256"],
                "training_seed": base_seed,
                "sparseseg_iteration_seeds": [
                    1_400_000 + base_seed + iteration for iteration in range(3)
                ],
            }
        )

    payload = {
        "schema_version": 3,
        "experiment": "formal 5 training-seed replicates x 3 fixed ROI budgets",
        "independent_volume": "HeLa2 COSEM volume",
        "pairing_definition": (
            "For a given trial and ROI budget every trainable method receives the same "
            "sparse-mask TIFF; trials 100-104 differ by training seed."
        ),
        "statistical_unit_warning": (
            "The released formal benchmark has 15 case rows but only three unique sparse "
            "masks. Trials 100-104 are training-seed replicates, not independent ROI-mask "
            "selections or independent biological volumes."
        ),
        "trial_ids": [100, 101, 102, 103, 104],
        "roi_budgets": [1, 5, 10],
        "unique_sparse_mask_count": 3,
        "input_contract": {
            "raw": "hela2_em_s3.tif",
            "dense_ground_truth_evaluation_only": "hela2_mito_s3.tif",
            "explicit_negative": "negative_hela2_em_s3.tif",
            "sparse_mask_template": "label_hela2_mito_{trial}_{roi}.tif",
        },
        "formal_input_provenance": {
            "dataset": "jrc_hela-2",
            "dataset_doi": "https://doi.org/10.25378/janelia.13114211",
            "viewer_url": "https://openorganelle.janelia.org/datasets/jrc_hela-2",
            "public_n5_root": "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5",
            "public_array_paths": {
                "raw": "em/fibsem-uint16/s3",
                "dense_ground_truth_evaluation_only": "labels/mito_seg/s3",
            },
            "public_to_tiff_transform": "transpose source axes by (1, 2, 0) to ZYX",
            "logical_sha256_c_order": {
                "raw": {
                    "dtype": "uint16",
                    "sha256": "1551fc1532e34aacba6cf7f3cf1b68bb473db1a4cdac74e668ce36e09192716a",
                },
                "dense_ground_truth_evaluation_only": {
                    "dtype": "uint8",
                    "sha256": "0f3e252e49c7a063227d0ab24d2cc5ab936f6189cedfe1a7b2d50490bf310d44",
                },
            },
            "explicit_negative_provenance": {
                "generator": "janelia_cosem/preprocessing.ijm (StackC.tif)",
                "exact_replay": "copy the author-generated mask from the accompanying Source Data",
                "validation": "shape is fixed; logical binary SHA-256 is printed by the formal runner",
            },
        },
        "seed_policies": {
            "SparseSeg": "1400000 + trial*100 + roi_num + iteration_zero_based",
            "Vanilla_UNet_and_nnUNet_controls": "trial*100 + roi_num",
            "MitoNet_sparse_adapted": 1337,
            "DeePiCt": 12345,
            "CellMap_COSEM_2D_3D": 42,
            "StarDist_formal_replay": "trial*100 + roi_num (legacy run did not record an explicit seed)",
        },
        "masks": masks,
        "cellmap_validity": {
            "asset": "masks/cellmap_validity_all_voxels.tif",
            "semantics": (
                "all voxels valid; this reproduces the native legacy CellMap baseline, "
                "where unlabeled voxels were treated as background"
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
