#!/usr/bin/env python3
"""Run three SparseSeg iterations with adaptive new2 on one Reviewer 3 case."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch


PAPER_EPOCHS = 60
PAPER_LOSS_WEIGHTS = [10.0, 0.1, 0.1, 0.05]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def completed_iteration_record(output_dir: Path, iteration_index: int):
    adaptive_root = (
        output_dir / "adaptive_iterated_mask" / f"iteration_{iteration_index}"
    )
    bridge_manifest = adaptive_root / "adaptive_iterated_mask_bridge_manifest.json"
    new2 = adaptive_root / "final" / "test_volume_label_new2.tif"
    complete = adaptive_root / "final" / "test_volume_label_save.tif"
    model = output_dir / f"model_{iteration_index}.pt"
    timing_path = output_dir / f"segment_cell_timing_iteration_{iteration_index}.json"
    required = (bridge_manifest, new2, complete, model, timing_path)
    if not all(path.is_file() for path in required):
        return None

    durable_path = (
        output_dir
        / f"adaptive_reviewer3_iteration_{iteration_index}_manifest.json"
    )
    if durable_path.is_file():
        record = json.loads(durable_path.read_text(encoding="utf-8"))
        if int(record.get("iteration_zero_based", -1)) != iteration_index:
            raise RuntimeError(f"invalid durable iteration record: {durable_path}")
        record["resume_source"] = "durable_iteration_manifest"
        return record

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    reconstructed_seconds = float(
        timing["training_wall_clock_seconds"]
        + timing["post_training_inference_new2_and_save_wall_clock_seconds"]
    )
    return {
        "iteration_zero_based": iteration_index,
        "seed": None,
        "wall_clock_seconds": reconstructed_seconds,
        "adaptive_bridge_manifest": str(bridge_manifest),
        "resume_source": "reconstructed_from_segment_cell_stage_timing",
    }


def restore_previous_complete_label(
    output_dir: Path, mask_name: str, iteration_index: int
) -> None:
    if iteration_index <= 0:
        return
    previous_complete = (
        output_dir
        / "adaptive_iterated_mask"
        / f"iteration_{iteration_index - 1}"
        / "final"
        / "test_volume_label_save.tif"
    )
    if not previous_complete.is_file():
        raise FileNotFoundError(previous_complete)
    complete = np.asarray(tifffile.imread(previous_complete) > 0, dtype=np.uint8)
    destination = output_dir / f"{mask_name}_new_base.tif"
    tifffile.imwrite(destination, complete, compression="zlib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--roi", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--continuous-selector", type=Path, required=True)
    parser.add_argument("--frozen-actions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=1400000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for required in (
        args.code_root,
        args.data_root,
        args.backend_dir,
        args.continuous_selector,
        args.frozen_actions,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    run_name = f"{args.trial}_{args.roi}"
    mask_name = f"label_hela2_mito_{run_name}"
    output_dir = args.output_root / run_name
    manifest_path = output_dir / "adaptive_segment_cell_run_manifest.json"
    if manifest_path.is_file():
        print(f"[resume] complete manifest exists: {manifest_path}", flush=True)
        return
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_records = []
    incomplete_seen = False
    for iteration_index in range(3):
        record = completed_iteration_record(output_dir, iteration_index)
        if record is None:
            incomplete_seen = True
        elif incomplete_seen:
            raise RuntimeError(
                "completed adaptive iterations are not a contiguous prefix"
            )
        else:
            completed_records.append(record)
    resume_start_iteration = len(completed_records)
    print(
        f"[resume] completed iteration prefix: {resume_start_iteration}/3",
        flush=True,
    )

    code_text = str(args.code_root.resolve())
    if code_text not in sys.path:
        sys.path.insert(0, code_text)
    import segment_cell_adaptive_v719 as segment_cell

    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    per_iteration = list(completed_records)
    for iteration_index in range(resume_start_iteration, 3):
        restore_previous_complete_label(
            output_dir, mask_name, iteration_index
        )
        iteration_seed = (
            int(args.seed_offset)
            + int(args.trial) * 100
            + int(args.roi)
            + iteration_index
        )
        set_seed(iteration_seed)
        iteration_started = time.perf_counter()
        segment_cell.main(
            interation_idx=iteration_index,
            filer_method=2,
            z_threshold=10,
            patch_scale=80,
            inference_stride=40,
            raw_name="hela2_em_s3",
            mask_name=mask_name,
            folder_name=str(output_dir),
            area_coef=1.0,
            edge_coef=1.0,
            iou_thresh=0.6,
            threshold=0.01,
            negative_threshold=3.0,
            low_weight_coeff=50.0,
            sparsity_weight=1.0,
            repeated_epoch=PAPER_EPOCHS,
            batch_size=12,
            num_samples=1000,
            thickness=2,
            base_folder=str(args.data_root),
            kernel_sizes=(3, 5, 7),
            Loss_list=PAPER_LOSS_WEIGHTS,
            if_Vit=False,
            refinement_profile="adaptive_iterated",
            evaluation_probability_quantile=98.95,
            adaptive_trial=args.trial,
            adaptive_run_name=run_name,
            adaptive_backend_dir=str(args.backend_dir),
            adaptive_continuous_selector=str(args.continuous_selector),
            adaptive_frozen_actions=str(args.frozen_actions),
            adaptive_sampling_policy="source_base85_850_120_30",
            adaptive_seed_offset=args.seed_offset,
        )
        torch.cuda.synchronize()
        iteration_record = {
            "iteration_zero_based": iteration_index,
            "seed": iteration_seed,
            "wall_clock_seconds": time.perf_counter() - iteration_started,
            "adaptive_bridge_manifest": str(
                output_dir
                / "adaptive_iterated_mask"
                / f"iteration_{iteration_index}"
                / "adaptive_iterated_mask_bridge_manifest.json"
            ),
            "resume_source": "completed_in_current_invocation",
        }
        per_iteration.append(iteration_record)
        iteration_record_path = (
            output_dir
            / f"adaptive_reviewer3_iteration_{iteration_index}_manifest.json"
        )
        iteration_record_path.write_text(
            json.dumps(iteration_record, indent=2), encoding="utf-8"
        )

    final_probability = output_dir / "edge_vol_probability_float32.tif"
    final_prediction = output_dir / "prediction_fixed_threshold.tif"
    final_checkpoint = output_dir / "model_2.pt"
    for required in (final_probability, final_prediction, final_checkpoint):
        if not required.is_file():
            raise RuntimeError(f"missing final output: {required}")
    manifest = {
        "schema_version": 800,
        "benchmark": "SparseSeg adaptive iterated-mask Reviewer 3 full grid",
        "iteration_level_resume_enabled": True,
        "iterations_executed_current_invocation": (
            3 - resume_start_iteration
        ),
        "trial": args.trial,
        "roi": args.roi,
        "run_name": run_name,
        "iterations_run": 3,
        "final_iteration_zero_based": 2,
        "dense_ground_truth_read": False,
        "edge_line_used_for_new2": False,
        "sampling_policy_after_iteration0": "source_base85_850_120_30",
        "paper_declared_parameters_changed": False,
        "paper_locked_parameters": {
            "architecture": "MultiKernelUNet",
            "kernel_sizes": [3, 5, 7],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs_per_iteration": PAPER_EPOCHS,
            "batch_size": 12,
            "loss_weights": PAPER_LOSS_WEIGHTS,
            "num_samples": 1000,
            "patch_size": 80,
            "inference_stride": 40,
            "thickness": 2,
        },
        "per_iteration": per_iteration,
        "total_end_to_end_wall_clock_seconds": float(
            sum(record["wall_clock_seconds"] for record in per_iteration)
        ),
        "wall_clock_aggregation": (
            "sum of the three persisted per-iteration end-to-end wall clocks; "
            "queue and interruption gaps excluded"
        ),
        "current_resume_invocation_wall_clock_seconds": (
            time.perf_counter() - started
        ),
        "resume_start_iteration_zero_based": resume_start_iteration,
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": __import__("os").environ.get("SLURM_ARRAY_TASK_ID"),
        "final_probability": str(final_probability),
        "final_probability_sha256": sha256(final_probability),
        "final_prediction_fixed_threshold": str(final_prediction),
        "final_prediction_fixed_threshold_sha256": sha256(final_prediction),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
