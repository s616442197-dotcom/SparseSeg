"""Headless MitoNet fine-tuning and whole-volume inference benchmark.

The runner mirrors the empanada-napari MitoNet fine-tune defaults without
requiring napari/Qt.  A run materializes the 200 xy slices, fine-tunes the
official MitoNet_v1 model on one sparse trial, predicts the complete volume,
and writes a binary TIFF plus an end-to-end timing record.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import tifffile
import torch
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_headless_napari_thread_worker() -> None:
    """Provide the one napari symbol used by empanada when Qt is absent."""
    try:
        from napari.qt.threading import thread_worker  # noqa: F401
        return
    except Exception:
        import sys
        import types

        napari_module = types.ModuleType("napari")
        qt_module = types.ModuleType("napari.qt")
        threading_module = types.ModuleType("napari.qt.threading")

        def thread_worker(function):
            return function

        threading_module.thread_worker = thread_worker
        qt_module.threading = threading_module
        napari_module.qt = qt_module
        sys.modules["napari"] = napari_module
        sys.modules["napari.qt"] = qt_module
        sys.modules["napari.qt.threading"] = threading_module


def stage_sparse_xy_dataset(raw: np.ndarray, sparse_mask: np.ndarray, root: Path) -> None:
    """Create empanada's ``source/images`` + ``source/masks`` dataset layout."""
    source = root / "sparse_trial"
    images_dir = source / "images"
    masks_dir = source / "masks"
    if root.exists():
        shutil.rmtree(root)
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for z_index, (image, mask) in enumerate(zip(raw, sparse_mask)):
        name = f"slice_{z_index:04d}.tif"
        tifffile.imwrite(images_dir / name, image.astype(np.uint8, copy=False))
        tifffile.imwrite(masks_dir / name, (mask > 0).astype(np.uint8, copy=False))


def make_finetune_config(
    *,
    base_config_path: Path,
    finetune_template_path: Path,
    base_model_path: Path,
    train_dir: Path,
    model_dir: Path,
    model_name: str,
    iterations: int,
    batch_size: int,
    workers: int,
    patch_size: int,
) -> Dict[str, object]:
    with base_config_path.open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    with finetune_template_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["MODEL"] = copy.deepcopy(model_config)
    config["FINETUNE"] = copy.deepcopy(model_config["FINETUNE"])
    config["MODEL"].pop("FINETUNE", None)
    config["MODEL"]["model"] = str(base_model_path)
    config["MODEL"]["model_quantized"] = None

    # Match the prior MitoNet benchmark: semantic fine-tuning of mitochondria.
    config["MODEL"]["thing_list"] = []
    config["FINETUNE"]["engine_params"]["thing_list"] = []
    config["FINETUNE"]["dataset_params"]["weight_gamma"] = 0.0

    train = config["TRAIN"]
    train["train_dir"] = str(train_dir)
    train["additional_train_dirs"] = None
    train["model_dir"] = str(model_dir)
    train["finetune_layer"] = "none"
    train["batch_size"] = batch_size
    train["workers"] = workers
    train["augmentations"][1]["min_height"] = patch_size
    train["augmentations"][1]["min_width"] = patch_size
    train["augmentations"][2]["height"] = patch_size
    train["augmentations"][2]["width"] = patch_size
    steps_per_epoch = max(1, 200 // batch_size)
    epochs = max(1, int(iterations // steps_per_epoch))
    train["schedule_params"]["epochs"] = epochs
    train["schedule_params"]["steps_per_epoch"] = steps_per_epoch
    train["save_freq"] = max(1, epochs // 5)
    train["metrics"][0]["labels"] = model_config["labels"]

    config["EVAL"]["eval_dir"] = None
    config["EVAL"]["epochs_per_eval"] = 1
    for metric in config["EVAL"]["metrics"]:
        metric["labels"] = model_config["labels"]
    config["model_name"] = model_name
    return config


def decode_semantic_tracker(trackers, shape) -> np.ndarray:
    """Decode MitoNet's xy tracker into the binary volume used by evaluation."""
    ensure_headless_napari_thread_worker()
    from empanada.inference.patterns import fill_volume, get_axis_trackers_by_class
    from empanada.inference.tracker import InstanceTracker
    from empanada_napari.inference import instance_relabel

    class_tracker = get_axis_trackers_by_class({"xy": trackers}, 1)[0]
    merged = InstanceTracker(1, class_tracker.label_divisor, shape, "xy")
    merged.instances = instance_relabel(class_tracker)
    decoded = np.zeros(shape, dtype=np.uint32)
    fill_volume(decoded, merged.instances)
    return (decoded > 0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--roi-num", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--finetune-template", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = f"{args.trial}_{args.roi_num}"
    output_dir = args.output_root / "mitonet_benchmark" / run_name
    prediction_path = output_dir / "masks" / "raw.tiff"
    timing_path = output_dir / "timing.json"
    if prediction_path.exists() and timing_path.exists() and not args.force:
        print(f"[resume] complete output exists: {prediction_path}")
        return

    start = time.perf_counter()
    start_utc = utc_now()
    stages: Dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_work = args.work_root / run_name
    data_dir = run_work / "train"
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage_start = time.perf_counter()
    raw_path = args.data_root / "hela2_em_s3.tif"
    label_path = args.data_root / f"label_hela2_mito_{args.trial}_{args.roi_num}.tif"
    raw = np.squeeze(tifffile.imread(raw_path))
    sparse_mask = np.squeeze(tifffile.imread(label_path))
    if raw.shape != sparse_mask.shape or raw.ndim != 3:
        raise ValueError(f"Input mismatch: raw={raw.shape}, label={sparse_mask.shape}")
    stage_sparse_xy_dataset(raw, sparse_mask, data_dir)
    stages["read_and_materialize_dataset_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    model_name = f"mitonet_hela2_mito_{run_name}"
    config = make_finetune_config(
        base_config_path=args.base_config,
        finetune_template_path=args.finetune_template,
        base_model_path=args.base_model,
        train_dir=data_dir,
        model_dir=model_dir,
        model_name=model_name,
        iterations=args.iterations,
        batch_size=args.batch_size,
        workers=args.workers,
        patch_size=args.patch_size,
    )
    config_snapshot = output_dir / "finetune_config_requested.yaml"
    with config_snapshot.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    from empanada_napari import finetune

    finetune.main(config)
    stages["finetune_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    trained_config_path = model_dir / f"{model_name}.yaml"
    with trained_config_path.open("r", encoding="utf-8") as handle:
        trained_config = yaml.safe_load(handle)
    ensure_headless_napari_thread_worker()
    from empanada_napari.inference import Engine3d

    engine = Engine3d(
        trained_config,
        inference_scale=1,
        label_divisor=10000,
        median_kernel_size=3,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.5,
        min_size=500,
        min_extent=5,
        semantic_only=True,
        use_gpu=True,
        save_panoptic=False,
    )
    _, trackers = engine.infer_on_axis(raw, "xy")
    prediction = decode_semantic_tracker(trackers, raw.shape)
    stages["whole_volume_inference_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(prediction_path, prediction, compression="zlib")
    stages["prediction_write_seconds"] = time.perf_counter() - stage_start

    end_to_end = time.perf_counter() - start
    timing = {
        "benchmark": "MitoNet",
        "trial": args.trial,
        "roi_num": args.roi_num,
        "measurement_status": "measured",
        "timing_scope": "raw/label read + dataset materialization + fine-tune + whole-volume inference + prediction TIFF write",
        "environment_and_pretrained_model_setup_included": False,
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "end_to_end_wall_clock_seconds": end_to_end,
        "stages": stages,
        "prediction_path": str(prediction_path),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(timing_path, timing)
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
