"""Run the official pretrained MitoNet_v1 once on the complete HeLa volume.

The result is intentionally independent of sparse trial/ROI selections. The
same prediction is evaluated against every paired trial to provide the dashed
pretrained reference requested by Reviewer 3 without counting inference time
15 times.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import tifffile
import torch
import yaml

from mitonet_benchmark import decode_semantic_tracker, ensure_headless_napari_thread_worker


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--raw-name", default="hela2_em_s3.tif")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / "mitonet_pretrained"
    prediction_path = output_dir / "masks" / "raw.tiff"
    timing_path = output_dir / "timing.json"
    if prediction_path.exists() and timing_path.exists() and not args.force:
        print(f"[resume] complete output exists: {prediction_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    process_start = time.perf_counter()
    start_utc = utc_now()
    stages: Dict[str, float] = {}

    stage_start = time.perf_counter()
    raw_path = args.data_root / args.raw_name
    raw = np.squeeze(tifffile.imread(raw_path))
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D raw volume, got {raw.shape}: {raw_path}")
    stages["raw_read_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    with args.base_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["model"] = str(args.base_model)
    config["model_quantized"] = None
    with (output_dir / "pretrained_inference_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    stages["config_materialization_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    ensure_headless_napari_thread_worker()
    from empanada_napari.inference import Engine3d

    engine = Engine3d(
        config,
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
    if prediction.shape != raw.shape:
        raise ValueError(f"Prediction mismatch: raw={raw.shape}, pred={prediction.shape}")
    stages["whole_volume_inference_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        prediction_path,
        prediction.astype(np.uint8, copy=False),
        photometric="minisblack",
        compression="zlib",
    )
    stages["prediction_write_seconds"] = time.perf_counter() - stage_start

    timing = {
        "schema_version": 1,
        "benchmark": "MitoNet-Pretrained",
        "implementation": "official MitoNet_v1 checkpoint; no sparse fine-tuning",
        "measurement_status": "measured",
        "timing_scope": "raw TIFF read + config materialization + whole-volume inference + compressed prediction TIFF write",
        "shared_across_trials_and_roi_budgets": True,
        "environment_and_model_download_setup_included": False,
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "end_to_end_wall_clock_seconds": time.perf_counter() - process_start,
        "stages": stages,
        "raw_path": str(raw_path),
        "base_model": str(args.base_model),
        "prediction_path": str(prediction_path),
        "predicted_foreground_fraction": float(prediction.mean()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with timing_path.open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
