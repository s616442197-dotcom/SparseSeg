#!/usr/bin/env python3
"""Run the archived 15-case/12-model benchmark with evaluator-compatible outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline_config.example.json"
MASK_ROOT = ROOT / "formal_assets" / "paired_roi_masks"
MASK_MANIFEST = MASK_ROOT / "fixed_paired_roi_masks.json"
DEFAULT_OUTPUT = ROOT / "formal_predictions"

SCHEDULE = {
    "SparseSeg": {"epochs": 60, "iterations": 3},
    "SparseSeg-ViT": {"epochs": 50, "iterations": 5},
    "StarDist": {"epochs": 50, "iterations": 1, "extra": ["--steps-per-epoch", "100"]},
    "MitoNet-Sparse-Finetuned": {"epochs": 8, "iterations": 1, "seed": 1337},
    "MitoNet-Pretrained": {"epochs": 1, "iterations": 1, "seed": 1337},
    "DeePict": {"epochs": 25, "iterations": 1, "seed": 12345},
    "COSEM-2D-UNet": {
        "epochs": 100,
        "iterations": 1,
        "seed": 42,
        "extra": ["--steps-per-epoch", "200"],
    },
    "COSEM-3D-UNet": {
        "epochs": 100,
        "iterations": 1,
        "seed": 42,
        "extra": ["--steps-per-epoch", "200"],
    },
    "Vanilla-UNet": {
        "epochs": 50,
        "iterations": 1,
        "extra": ["--num-samples", "1000", "--batch-size", "12"],
    },
    "nnU-Net": {"epochs": 50, "iterations": 1},
    "Vanilla-UNet-SparseMatched": {
        "epochs": 50,
        "iterations": 1,
        "extra": ["--num-samples", "1000", "--batch-size", "12"],
    },
    "nnU-Net-SparseMatched": {"epochs": 50, "iterations": 1},
}


def expand(value: object, *, allow_unresolved: bool = False) -> str:
    raw = str(value)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    unresolved = "${" in expanded or ("%" in expanded and expanded.count("%") >= 2)
    if unresolved and not allow_unresolved:
        raise RuntimeError(f"unresolved environment variable: {raw}")
    return expanded


def slug(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def logical_sha256(array) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_models(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("config must contain a models list")
    names = [str(item["name"]) for item in models]
    if set(names) != set(SCHEDULE):
        raise ValueError(f"config/model mismatch: {sorted(set(SCHEDULE) ^ set(names))}")
    return models


def output_path(root: Path, model: str, trial: int, roi: int) -> Path:
    case = f"{trial}_{roi}"
    stem = f"prediction_hela2_mito_{case}.tif"
    mapping = {
        "SparseSeg": root / "response" / "sparseseg_adaptive_iterated_v735" / case / stem,
        "SparseSeg-ViT": root / "sparseseg_vit" / case / stem,
        "StarDist": root / "stardist" / stem,
        "MitoNet-Sparse-Finetuned": root / "mitonet_benchmark" / case / "masks" / "raw.tiff",
        "MitoNet-Pretrained": root / "response" / "mitonet_pretrained" / "masks" / "raw.tiff",
        "DeePict": root / "data" / "deepict_finetune" / "tif_predictions" / f"deepict_hela2_mito_{case}_whole_volume.tif",
        "COSEM-2D-UNet": root / "data" / "cellmap_official_2d_tif_predictions" / f"cellmap_2d_unet_hela2_mito_{case}_whole_volume.tif",
        "COSEM-3D-UNet": root / "data" / "cellmap_official_3d_tif_predictions" / f"cellmap_3d_unet_hela2_mito_{case}_whole_volume.tif",
        "Vanilla-UNet": root / "vanilla_unet" / case / stem,
        "nnU-Net": root / "nnUNetv2_official_raw" / case / stem,
        "Vanilla-UNet-SparseMatched": root / "vanilla_unet_sparse_matched" / case / stem,
        "nnU-Net-SparseMatched": root / "nnUNetv2_official_sparse_matched" / case / stem,
    }
    return mapping[model]


def build_command(
    item: dict[str, object],
    *,
    raw: Path,
    sparse: Path,
    negative: Path,
    output: Path,
    work: Path,
    trial: int,
    roi: int,
    seed: int,
    device: str | None,
    dry_run: bool = False,
) -> list[str]:
    name = str(item["name"])
    schedule = SCHEDULE[name]
    interpreter = Path(expand(item["python"], allow_unresolved=dry_run))
    adapter = ROOT / str(item["adapter"])
    if not dry_run and not interpreter.is_file():
        raise FileNotFoundError(f"{name} interpreter not found: {interpreter}")
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    command = [str(interpreter), str(adapter)]
    command.extend(
        expand(value, allow_unresolved=dry_run) for value in item.get("args", [])
    )
    command.extend(
        [
            "--raw", str(raw),
            "--sparse-label", str(sparse),
            "--negative-label", str(negative),
            "--output", str(output),
            "--work-dir", str(work),
            "--epochs", str(schedule["epochs"]),
            "--iterations", str(schedule["iterations"]),
            "--trial", str(trial),
            "--roi-num", str(roi),
            "--seed", str(schedule.get("seed", seed)),
            "--device", str(device or item.get("device", "cuda")),
        ]
    )
    command.extend(str(value) for value in schedule.get("extra", []))
    return command


def metrics(model: str, trial: int, roi: int, prediction: Path, gt, elapsed: float) -> dict[str, object]:
    import numpy as np
    import tifffile

    pred = np.squeeze(np.asarray(tifffile.imread(prediction))) > 0
    if pred.shape != gt.shape:
        raise ValueError(f"{model} {trial}_{roi}: {pred.shape} != {gt.shape}")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return {
        "model": model,
        "trial": trial,
        "roi_num": roi,
        "absolute_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_foreground_fraction": float(pred.mean()),
        "gt_foreground_fraction": float(gt.mean()),
        "end_to_end_wall_clock_seconds": elapsed,
        "prediction": str(prediction.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--cases", nargs="*", help="Subset such as 100_1 104_10")
    parser.add_argument("--device")
    parser.add_argument("--install-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MASK_MANIFEST.read_text(encoding="utf-8"))
    if args.install_masks:
        subprocess.run(
            [sys.executable, str(MASK_ROOT / "install_formal_masks.py"), "--data-root", str(args.data_root)],
            check=True,
        )
    contract = manifest["input_contract"]
    raw = args.data_root / contract["raw"]
    gt_path = args.data_root / contract["dense_ground_truth_evaluation_only"]
    negative = args.data_root / contract["explicit_negative"]
    required = [raw, gt_path, negative]
    selected_cases = set(args.cases or [row["case_id"] for row in manifest["rows"]])
    rows = [row for row in manifest["rows"] if row["case_id"] in selected_cases]
    if {row["case_id"] for row in rows} != selected_cases:
        raise ValueError(f"unknown --cases: {sorted(selected_cases - {row['case_id'] for row in rows})}")
    required.extend(args.data_root / row["installed_sparse_filename"] for row in rows)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal inputs are incomplete. Put files under --data-root using the exact "
            f"names in fixed_paired_roi_masks.json. Missing: {missing}"
        )
    import numpy as np
    import tifffile

    shapes = {}
    for current in required:
        with tifffile.TiffFile(current) as handle:
            shapes[str(current)] = tuple(handle.series[0].shape)
    expected_shape = tuple(manifest["masks"]["1"]["shape_zyx"])
    wrong_shapes = {name: shape for name, shape in shapes.items() if shape != expected_shape}
    if wrong_shapes:
        raise ValueError(f"formal TIFF shape mismatch; expected {expected_shape}: {wrong_shapes}")
    public_inputs = manifest["formal_input_provenance"]["logical_sha256_c_order"]
    gt_array = None
    for contract_key, specification in public_inputs.items():
        current = args.data_root / contract[contract_key]
        array = np.squeeze(np.asarray(tifffile.imread(current)))
        actual_dtype = str(array.dtype)
        actual_hash = logical_sha256(array)
        if actual_dtype != specification["dtype"] or actual_hash != specification["sha256"]:
            raise ValueError(
                f"formal public input mismatch for {current}: "
                f"dtype={actual_dtype}, logical_sha256={actual_hash}; "
                f"expected dtype={specification['dtype']}, "
                f"logical_sha256={specification['sha256']}"
            )
        if contract_key == "dense_ground_truth_evaluation_only":
            gt_array = array
    negative_array = np.squeeze(np.asarray(tifffile.imread(negative)))
    negative_hash = logical_sha256((negative_array > 0).astype(np.uint8))
    logical_mismatches = {}
    for row in rows:
        current = args.data_root / row["installed_sparse_filename"]
        binary = np.squeeze(np.asarray(tifffile.imread(current))) > 0
        actual = hashlib.sha256(
            binary.astype(np.uint8).tobytes(order="C")
        ).hexdigest()
        expected = manifest["masks"][str(row["roi_num"])][
            "logical_uint8_sha256"
        ]
        if actual != expected:
            logical_mismatches[str(current)] = actual
    if logical_mismatches:
        raise ValueError(
            "formal sparse-mask content hash mismatch: "
            f"{logical_mismatches}"
        )
    if args.validate_only:
        print(
            f"validated {len(rows)} formal cases under {args.data_root}; "
            f"shape={expected_shape}; public raw/GT hashes=OK; "
            f"fixed-mask hashes=OK; negative logical_binary_sha256={negative_hash}"
        )
        return

    if gt_array is None:
        raise RuntimeError("dense ground truth was not validated")
    gt = gt_array > 0
    models = load_models(args.config.resolve())
    selected_models = set(args.models or [str(item["name"]) for item in models])
    unknown_models = selected_models - {str(item["name"]) for item in models}
    if unknown_models:
        raise ValueError(f"unknown --models: {sorted(unknown_models)}")
    results: list[dict[str, object]] = []
    planned_commands = 0
    for item in models:
        model = str(item["name"])
        if model not in selected_models:
            continue
        for row in rows:
            trial, roi = int(row["trial"]), int(row["roi_num"])
            prediction = output_path(args.output_root, model, trial, roi)
            work = args.output_root / "work" / slug(model) / row["case_id"]
            if not args.dry_run:
                prediction.parent.mkdir(parents=True, exist_ok=True)
            command = build_command(
                item,
                raw=raw,
                sparse=args.data_root / row["installed_sparse_filename"],
                negative=negative,
                output=prediction,
                work=work,
                trial=trial,
                roi=roi,
                seed=int(row["training_seed"]),
                device=args.device,
                dry_run=args.dry_run,
            )
            print(f"[{model} {row['case_id']}] {subprocess.list2cmdline(command)}", flush=True)
            planned_commands += 1
            if args.dry_run:
                continue
            started = time.perf_counter()
            if not prediction.is_file() or args.overwrite:
                subprocess.run(command, check=True, cwd=ROOT)
            timing = prediction.with_suffix(".timing.json")
            elapsed = time.perf_counter() - started
            if timing.is_file():
                elapsed = float(json.loads(timing.read_text(encoding="utf-8"))["wall_clock_seconds"])
            if not prediction.is_file():
                raise FileNotFoundError(prediction)
            results.append(metrics(model, trial, roi, prediction, gt, elapsed))
    if args.dry_run:
        print(
            f"expanded {planned_commands} commands "
            f"({len(selected_models)} models x {len(rows)} cases)"
        )
        return
    target = args.output_root / "formal_evaluation_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} metric rows to {target}")
    print(
        "Evaluator-compatible layout is ready. Run evaluation_cross_trials_extreme.py "
        f"with --empanda-root {args.output_root} --strict."
    )


if __name__ == "__main__":
    main()
