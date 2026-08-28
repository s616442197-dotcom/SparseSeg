#!/usr/bin/env python3
"""Prepare official nnU-Net v2 Tiff3DIO datasets for the 15 sparse trials.

The raw and sparse-matched conditions use separate dataset IDs but share the
same five shifted EM channels. No network or planner code is implemented here;
the generated folders are consumed by the pip-installed nnU-Net v2 commands.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt


TRIALS = (100, 101, 102, 103, 104)
ROI_NUMS = (1, 5, 10)
RAW_DATASET_BASE = 701
MATCHED_DATASET_BASE = 751
SPACING = (1.0, 1.0, 1.0)
SHIFTS = (-2, -1, 0, 1, 2)
SOFT_CODE_OFFSET = 3
SOFT_CODE_SCALE = 2500.0
RELIABLE_NEGATIVE_CODE = 2


def task_to_trial_roi(task_id: int) -> tuple[int, int]:
    if not 0 <= task_id < 15:
        raise ValueError(f"task-id must be 0..14, got {task_id}")
    return TRIALS[task_id % 5], ROI_NUMS[task_id // 5]


def dataset_id(variant: str, task_id: int) -> int:
    return (RAW_DATASET_BASE if variant == "raw" else MATCHED_DATASET_BASE) + task_id


def dataset_name(variant: str, task_id: int) -> str:
    trial, roi_num = task_to_trial_roi(task_id)
    suffix = "Raw" if variant == "raw" else "SparseMatched"
    return f"Dataset{dataset_id(variant, task_id):03d}_VEMHela2MitoT{trial}R{roi_num}{suffix}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_spacing(path: Path) -> None:
    atomic_json(path, {"spacing": list(SPACING)})


def link_or_validate(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Unexpected symlink target: {destination} -> {destination.resolve()}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {destination}")
    destination.symlink_to(source)


def write_tiff_atomic(path: Path, array: np.ndarray, *, compression: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    tifffile.imwrite(
        temporary,
        array,
        photometric="minisblack",
        compression=compression,
        bigtiff=array.nbytes >= 3_500_000_000,
    )
    os.replace(temporary, path)


def prepare_common(source_root: Path, work_root: Path) -> None:
    started = time.perf_counter()
    raw_path = source_root / "hela2_em_s3.tif"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    shared = work_root / "shared_raw"
    shared.mkdir(parents=True, exist_ok=True)
    raw = np.squeeze(tifffile.imread(raw_path))
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D raw volume, got {raw.shape}")
    z_indices = np.arange(raw.shape[0])
    outputs = []
    for channel, shift in enumerate(SHIFTS):
        output = shared / f"hela2_em_shift_{shift:+d}.tif"
        if shift == 0:
            link_or_validate(raw_path, output)
        elif not output.exists():
            shifted = np.ascontiguousarray(raw[np.clip(z_indices + shift, 0, raw.shape[0] - 1)])
            write_tiff_atomic(output, shifted, compression=None)
            del shifted
        write_spacing(output.with_suffix(".json"))
        outputs.append({
            "channel": channel,
            "z_shift": shift,
            "path": str(output),
            "bytes": output.stat().st_size,
        })
    atomic_json(work_root / "common_manifest.json", {
        "schema_version": 1,
        "source_raw": str(raw_path),
        "source_shape": list(raw.shape),
        "source_dtype": str(raw.dtype),
        "spacing": list(SPACING),
        "channels": outputs,
        "wall_clock_seconds": time.perf_counter() - started,
    })


def build_soft_negative(mask: np.ndarray, radius: float = 50.0) -> np.ndarray:
    distance = distance_transform_edt(1 - mask.astype(np.uint8))
    k = radius / 6.0
    soft = 1.0 / (1.0 + np.exp(-(distance - radius) / k))
    soft = np.clip(soft - 0.01, 0.0, None)
    return (0.1 * soft).astype(np.float32)


def encode_matched_target(mask: np.ndarray, reliable_negative: np.ndarray) -> np.ndarray:
    soft = build_soft_negative(mask)
    encoded = np.zeros(mask.shape, dtype=np.uint8)
    soft_nonzero = soft > 0
    encoded[soft_nonzero] = np.clip(
        SOFT_CODE_OFFSET + np.rint(soft[soft_nonzero] * SOFT_CODE_SCALE),
        SOFT_CODE_OFFSET,
        253,
    ).astype(np.uint8)
    encoded[reliable_negative > 0] = RELIABLE_NEGATIVE_CODE
    encoded[mask > 0] = 1
    return encoded


def prepare_dataset(
    variant: str,
    task_id: int,
    source_root: Path,
    work_root: Path,
    nnunet_raw: Path,
) -> None:
    started = time.perf_counter()
    trial, roi_num = task_to_trial_roi(task_id)
    name = dataset_name(variant, task_id)
    folder = nnunet_raw / name
    images_tr = folder / "imagesTr"
    images_ts = folder / "imagesTs"
    labels_tr = folder / "labelsTr"
    for directory in (images_tr, images_ts, labels_tr):
        directory.mkdir(parents=True, exist_ok=True)

    case_id = f"hela2_t{trial}_r{roi_num}"
    shared = work_root / "shared_raw"
    for channel, shift in enumerate(SHIFTS):
        source = shared / f"hela2_em_shift_{shift:+d}.tif"
        if not source.exists():
            raise FileNotFoundError(f"Run prepare-common first: {source}")
        for image_folder in (images_tr, images_ts):
            destination = image_folder / f"{case_id}_{channel:04d}.tif"
            link_or_validate(source, destination)
    # Tiff3DIO expects one case-level sidecar (case.json), not one sidecar per
    # channel (case_0000.json). The same spacing applies to every channel.
    for image_folder in (images_tr, images_ts):
        write_spacing(image_folder / f"{case_id}.json")

    sparse_path = source_root / f"label_hela2_mito_{trial}_{roi_num}.tif"
    sparse = (np.squeeze(tifffile.imread(sparse_path)) > 0).astype(np.uint8)
    if variant == "raw":
        target = sparse
        labels = {"background": 0, "mitochondria": 1}
        encoding = "binary sparse positive label; unlabeled voxels treated as background by official loss"
    else:
        negative_path = source_root / "negative_hela2_em_s3.tif"
        reliable_negative = np.squeeze(tifffile.imread(negative_path)) > 0
        if reliable_negative.shape != sparse.shape:
            raise ValueError(
                f"negative/sparse shape mismatch: {reliable_negative.shape} versus {sparse.shape}"
            )
        target = encode_matched_target(sparse, reliable_negative)
        labels = {"background": 0, "mitochondria": 1, "ignore": RELIABLE_NEGATIVE_CODE}
        encoding = (
            "1=positive, 2=reliable negative, 3..253=quantized full-volume soft-negative weight; "
            "decoded only by nnUNetTrainerVEMSparseMatched50epochs"
        )

    label_path = labels_tr / f"{case_id}.tif"
    if not label_path.exists():
        write_tiff_atomic(label_path, target, compression="zlib")
    write_spacing(label_path.with_suffix(".json"))

    dataset_json = {
        "channel_names": {str(i): f"EM_z_shift_{shift:+d}" for i, shift in enumerate(SHIFTS)},
        "labels": labels,
        "numTraining": 1,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "Tiff3DIO",
        "name": name,
        "description": (
            "Official nnU-Net v2 2D benchmark on one five-channel 2.5D EM volume. "
            f"Condition={variant}, trial={trial}, roi={roi_num}."
        ),
        "reference": "SparseSeg reviewer-response controlled benchmark",
        "licence": "Research use; source data provenance retained by the project",
        "converted_by": "prepare_official_nnunetv2_data.py",
    }
    atomic_json(folder / "dataset.json", dataset_json)
    values, counts = np.unique(target, return_counts=True)
    atomic_json(folder / "dataset_manifest.json", {
        "schema_version": 1,
        "variant": variant,
        "dataset_id": dataset_id(variant, task_id),
        "dataset_name": name,
        "task_id": task_id,
        "trial": trial,
        "roi_num": roi_num,
        "case_id": case_id,
        "source_sparse_label": str(sparse_path),
        "target_encoding": encoding,
        "target_shape": list(target.shape),
        "target_code_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "positive_voxels": int(np.count_nonzero(sparse)),
        "wall_clock_seconds": time.perf_counter() - started,
    })


def validate_dataset(variant: str, task_id: int, nnunet_raw: Path) -> None:
    name = dataset_name(variant, task_id)
    folder = nnunet_raw / name
    dataset_json = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    trial, roi_num = task_to_trial_roi(task_id)
    case_id = f"hela2_t{trial}_r{roi_num}"
    images = [folder / "imagesTr" / f"{case_id}_{i:04d}.tif" for i in range(5)]
    label = folder / "labelsTr" / f"{case_id}.tif"
    missing = [str(p) for p in (*images, label) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))
    image_shapes = [tuple(tifffile.TiffFile(p).series[0].shape) for p in images]
    label_shape = tuple(tifffile.TiffFile(label).series[0].shape)
    if len(set(image_shapes + [label_shape])) != 1:
        raise ValueError(f"Geometry mismatch: images={image_shapes}, label={label_shape}")
    if dataset_json["numTraining"] != 1 or dataset_json["file_ending"] != ".tif":
        raise ValueError(f"Unexpected dataset.json: {dataset_json}")
    print(json.dumps({
        "status": "valid",
        "variant": variant,
        "dataset_id": dataset_id(variant, task_id),
        "dataset_name": name,
        "shape": label_shape,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = subparsers.add_parser("prepare-common")
    common.add_argument("--source-root", type=Path, required=True)
    common.add_argument("--work-root", type=Path, required=True)
    for command in ("prepare-dataset", "validate-dataset"):
        current = subparsers.add_parser(command)
        current.add_argument("--variant", choices=("raw", "sparse_matched"), required=True)
        current.add_argument("--task-id", type=int, required=True)
        current.add_argument("--nnunet-raw", type=Path, required=True)
        if command == "prepare-dataset":
            current.add_argument("--source-root", type=Path, required=True)
            current.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-common":
        prepare_common(args.source_root.resolve(), args.work_root.resolve())
    elif args.command == "prepare-dataset":
        prepare_dataset(
            args.variant,
            args.task_id,
            args.source_root.resolve(),
            args.work_root.resolve(),
            args.nnunet_raw.resolve(),
        )
    else:
        validate_dataset(args.variant, args.task_id, args.nnunet_raw.resolve())


if __name__ == "__main__":
    main()
