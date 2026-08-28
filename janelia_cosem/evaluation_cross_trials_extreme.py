"""Evaluate extreme-sparse cross-trial predictions with absolute voxel metrics.

Outputs one row per model/trial/ROI budget with absolute metrics and the
historical comparison-set-dependent log-normalized relative IoU. Binary-mask
predictions are read with ``pred > 0``. SparseSeg RGB
predictions retain their historical channel/percentile loader so the new table
is directly comparable with the previous evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
    import tifffile
except ModuleNotFoundError as dependency_error:
    raise RuntimeError(
        "evaluation_cross_trials_extreme.py requires numpy and tifffile. "
        "Install the benchmark_pipeline requirements in the active environment."
    ) from dependency_error

TRIALS = (100, 101, 102, 103, 104)
ROI_NUMS = (1, 5, 10)
NON_DEGENERATE_MIN_FOREGROUND_RATIO = 0.01
NON_DEGENERATE_MAX_FOREGROUND_RATIO = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_empanda_root() -> Path:
    return Path(__file__).resolve().parent / "benchmark_pipeline" / "formal_predictions"


def load_sparse_seg_pred(
    path: Path,
    *,
    percentile_fraction: float = 0.95,
    minimum_ratio: float = 0.3,
    channel: int = 1,
) -> np.ndarray:
    prediction = np.squeeze(tifffile.imread(path))
    if prediction.ndim == 4:
        if not 0 <= channel < prediction.shape[-1]:
            raise ValueError(f"Channel {channel} is invalid for {path}: {prediction.shape}")
        prediction = prediction[..., channel]
    elif prediction.ndim != 3:
        raise ValueError(f"Unexpected SparseSeg shape for {path}: {prediction.shape}")
    percentile_threshold = np.percentile(prediction, 100.0 * percentile_fraction)
    dtype_scale = 255.0 if np.issubdtype(prediction.dtype, np.integer) else 1.0
    threshold = max(float(percentile_threshold), minimum_ratio * dtype_scale)
    return prediction >= threshold


def load_binary_mask_pred(path: Path) -> np.ndarray:
    prediction = np.squeeze(tifffile.imread(path))
    if prediction.ndim != 3:
        raise ValueError(f"Expected a 3D mask for {path}, got {prediction.shape}")
    return prediction > 0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    templates: Tuple[str, ...]
    loader: Callable[..., np.ndarray]
    loader_kwargs: Mapping[str, object]
    default_enabled: bool = True


def build_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            "SparseSeg-Legacy",
            ("{repo}/label_{celltype}_{organelle}_{trial}_{roi}_control/volume_mask_pred.tiff",),
            load_sparse_seg_pred,
            {"percentile_fraction": 0.95, "minimum_ratio": 0.3, "channel": 1},
            False,
        ),
        ModelSpec(
            "SparseSeg-Tuned-v1",
            ("{empanda}/response/sparseseg_tuned_v1/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
            False,
        ),
        ModelSpec(
            "SparseSeg-Tuned",
            ("{empanda}/response/sparseseg_tuned/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
            False,
        ),
        ModelSpec(
            "SparseSeg-Optimized-Iterative",
            (
                "{empanda}/response/sparseseg_optimized_iterative_v52/"
                "{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",
            ),
            load_binary_mask_pred,
            {},
            False,
        ),
        ModelSpec(
            "SparseSeg",
            (
                "{empanda}/response/sparseseg_adaptive_iterated_v735/"
                "{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",
                "{empanda}/response/sparseseg_balanced_final_v25/"
                "{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",
            ),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "SparseSeg-Geometry-v5",
            ("{empanda}/response/sparseseg_geometry_v5/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
            False,
        ),
        ModelSpec(
            "SparseSeg-ViT",
            ("{empanda}/sparseseg_vit/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_sparse_seg_pred,
            {"percentile_fraction": 0.95, "minimum_ratio": 0.3, "channel": 1},
        ),
        ModelSpec(
            "SparseSeg-Backbone-Conventional",
            ("{empanda}/response/sparseseg_backbone_conventional/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
            False,
        ),
        ModelSpec(
            "StarDist",
            ("{empanda}/stardist/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "MitoNet-Sparse-Finetuned",
            (
                "{empanda}/mitonet_benchmark/{trial}_{roi}/masks/raw.tiff",
                "{empanda}/mitonet/result/{trial}_{roi}/masks/raw.tiff",
            ),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "MitoNet-Pretrained",
            ("{empanda}/response/mitonet_pretrained/masks/raw.tiff",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "DeePict",
            ("{empanda}/data/deepict_finetune/tif_predictions/deepict_{celltype}_{organelle}_{trial}_{roi}_whole_volume.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "COSEM-2D-UNet",
            ("{empanda}/data/cellmap_official_2d_tif_predictions/cellmap_2d_unet_{celltype}_{organelle}_{trial}_{roi}_whole_volume.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "COSEM-3D-UNet",
            ("{empanda}/data/cellmap_official_3d_tif_predictions/cellmap_3d_unet_{celltype}_{organelle}_{trial}_{roi}_whole_volume.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "Vanilla-UNet",
            ("{empanda}/vanilla_unet/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "nnU-Net",
            (
                "{empanda}/nnUNetv2_official_raw/{trial}_{roi}/"
                "prediction_{celltype}_{organelle}_{trial}_{roi}.tif",
            ),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "Vanilla-UNet-SparseMatched",
            ("{empanda}/vanilla_unet_sparse_matched/{trial}_{roi}/prediction_{celltype}_{organelle}_{trial}_{roi}.tif",),
            load_binary_mask_pred,
            {},
        ),
        ModelSpec(
            "nnU-Net-SparseMatched",
            (
                "{empanda}/nnUNetv2_official_sparse_matched/{trial}_{roi}/"
                "prediction_{celltype}_{organelle}_{trial}_{roi}.tif",
            ),
            load_binary_mask_pred,
            {},
        ),
    ]


def resolve_prediction(
    spec: ModelSpec,
    *,
    repo: Path,
    empanda: Path,
    celltype: str,
    organelle: str,
    trial: int,
    roi: int,
) -> Tuple[Optional[Path], List[Path]]:
    candidates = [
        Path(
            template.format(
                repo=repo,
                empanda=empanda,
                celltype=celltype,
                organelle=organelle,
                trial=trial,
                roi=roi,
            )
        )
        for template in spec.templates
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def absolute_metrics(gt: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    gt_bool = np.asarray(gt, dtype=bool)
    pred_bool = np.asarray(prediction, dtype=bool)
    if gt_bool.shape != pred_bool.shape:
        raise ValueError(f"Shape mismatch: gt={gt_bool.shape}, prediction={pred_bool.shape}")
    total = int(gt_bool.size)
    gt_positive = int(np.count_nonzero(gt_bool))
    pred_positive = int(np.count_nonzero(pred_bool))
    true_positive = int(np.count_nonzero(pred_bool[gt_bool]))
    false_positive = pred_positive - true_positive
    false_negative = gt_positive - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    predicted_fraction = safe_divide(pred_positive, total)
    gt_fraction = safe_divide(gt_positive, total)
    foreground_ratio = predicted_fraction / gt_fraction if gt_fraction else math.inf
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "gt_foreground_voxels": gt_positive,
        "predicted_foreground_voxels": pred_positive,
        "total_voxels": total,
        "absolute_iou": safe_divide(
            true_positive, true_positive + false_positive + false_negative
        ),
        "precision": safe_divide(true_positive, true_positive + false_positive),
        "recall": safe_divide(true_positive, true_positive + false_negative),
        "predicted_foreground_fraction": predicted_fraction,
        "gt_foreground_fraction": gt_fraction,
        "foreground_fraction_absolute_error": abs(predicted_fraction - gt_fraction),
        "predicted_to_gt_foreground_ratio": foreground_ratio,
        "constant_prediction": pred_positive in (0, total),
        "degenerate_prediction": (
            pred_positive in (0, total)
            or foreground_ratio < NON_DEGENERATE_MIN_FOREGROUND_RATIO
            or foreground_ratio > NON_DEGENERATE_MAX_FOREGROUND_RATIO
        ),
    }


def add_relative_iou(rows: Sequence[Dict[str, object]]) -> float:
    """Add the historical comparison-set-dependent log-normalized IoU."""
    transformed = [math.log1p(100.0 * float(row["absolute_iou"])) for row in rows]
    denominator = max(transformed, default=0.0)
    for row, value in zip(rows, transformed):
        row["relative_iou_log_normalized"] = (
            value / denominator if denominator > 0.0 else 0.0
        )
    return denominator


def read_timing(prediction_path: Path) -> Dict[str, object]:
    candidates = [
        prediction_path.with_suffix(".timing.json"),
        prediction_path.parent / "timing.json",
        prediction_path.parent / "run_manifest.json",
        prediction_path.parent / "next_iteration_manifest.json",
        prediction_path.parent.parent / "timing.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            timing = json.load(handle)
        duration = timing.get("end_to_end_wall_clock_seconds")
        if duration is None:
            duration = timing.get("total_end_to_end_wall_clock_seconds")
        if duration is None:
            duration = timing.get("complete_second_iteration_wall_clock_seconds")
        if duration is None:
            duration = timing.get("wall_clock_seconds")
        shared_duration = timing.get("shared_raw_to_feature_preprocessing_seconds")
        shared_identity = timing.get("shared_preprocessing_timing_path", "")
        if shared_duration is not None and not shared_identity:
            shared_identity = "SparseSeg-Shared-Feature-Preprocessing"
        return {
            "timing_status": timing.get("measurement_status", "measured"),
            "end_to_end_wall_clock_seconds": float(duration) if duration is not None else None,
            "timing_path": str(candidate),
            "shared_preprocessing_wall_clock_seconds": (
                float(shared_duration) if shared_duration is not None else None
            ),
            "shared_preprocessing_timing_path": str(shared_identity),
        }
    return {
        "timing_status": "unavailable_existing_artifact",
        "end_to_end_wall_clock_seconds": None,
        "timing_path": "",
        "shared_preprocessing_wall_clock_seconds": None,
        "shared_preprocessing_timing_path": "",
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: Iterable[Mapping[str, object]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def unique_timing_values(
    rows: Iterable[Mapping[str, object]],
    *,
    duration_key: str,
    identity_key: str,
) -> List[float]:
    values: Dict[str, float] = {}
    for row in rows:
        duration = row.get(duration_key)
        identity = str(row.get(identity_key, ""))
        if duration is None or not identity:
            continue
        duration_f = float(duration)
        if math.isfinite(duration_f):
            values.setdefault(identity, duration_f)
    return list(values.values())


def summarize_group(rows: Sequence[Mapping[str, object]], *, model: str, roi: object) -> Dict[str, object]:
    result: Dict[str, object] = {
        "model": model,
        "roi_num": roi,
        "completed_trials": len(rows),
    }
    for key in (
        "absolute_iou",
        "relative_iou_log_normalized",
        "precision",
        "recall",
        "predicted_foreground_fraction",
        "foreground_fraction_absolute_error",
        "predicted_to_gt_foreground_ratio",
    ):
        values = finite_values(rows, key)
        result[f"mean_{key}"] = statistics.fmean(values) if values else None
        result[f"std_{key}"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
    result["constant_prediction_trials"] = sum(
        bool(row.get("constant_prediction")) for row in rows
    )
    result["degenerate_prediction_trials"] = sum(
        bool(row.get("degenerate_prediction")) for row in rows
    )
    mean_ratio = result.get("mean_predicted_to_gt_foreground_ratio")
    result["eligible_non_degenerate_summary"] = bool(
        mean_ratio is not None
        and NON_DEGENERATE_MIN_FOREGROUND_RATIO
        <= float(mean_ratio)
        <= NON_DEGENERATE_MAX_FOREGROUND_RATIO
    )
    timing_rows = finite_values(rows, "end_to_end_wall_clock_seconds")
    unique_runs = unique_timing_values(
        rows,
        duration_key="end_to_end_wall_clock_seconds",
        identity_key="timing_path",
    )
    shared_runs = unique_timing_values(
        rows,
        duration_key="shared_preprocessing_wall_clock_seconds",
        identity_key="shared_preprocessing_timing_path",
    )
    result["measured_timing_trials"] = len(timing_rows)
    result["unique_measured_timing_runs"] = len(unique_runs)
    result["unmeasured_timing_trials"] = len(rows) - len(timing_rows)
    result["shared_preprocessing_runs"] = len(shared_runs)
    result["total_end_to_end_wall_clock_seconds"] = (
        sum(unique_runs) + sum(shared_runs) if unique_runs or shared_runs else None
    )
    result["mean_end_to_end_wall_clock_seconds"] = (
        statistics.fmean(timing_rows) if timing_rows else None
    )
    return result


def make_summaries(rows: Sequence[Mapping[str, object]]):
    by_roi = []
    model_names = sorted({str(row["model"]) for row in rows})
    for model in model_names:
        for roi in ROI_NUMS:
            group = [row for row in rows if row["model"] == model and int(row["roi_num"]) == roi]
            if group:
                by_roi.append(summarize_group(group, model=model, roi=roi))
    overall = []
    for model in model_names:
        group = [row for row in rows if row["model"] == model]
        overall.append(summarize_group(group, model=model, roi="all"))
    return by_roi, overall


def componentwise_gates(summary_overall: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    def build(candidates: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        definitions = {
            "absolute_iou": ("mean_absolute_iou", max),
            "precision": ("mean_precision", max),
            "recall": ("mean_recall", max),
            "foreground_fraction_absolute_error": (
                "mean_foreground_fraction_absolute_error",
                min,
            ),
        }
        result = {}
        for label, (key, reducer) in definitions.items():
            usable = [row for row in candidates if row.get(key) is not None]
            if not usable:
                result[label] = None
                continue
            value = reducer(float(row[key]) for row in usable)
            winners = [str(row["model"]) for row in usable if float(row[key]) == value]
            result[label] = {"value": value, "models": winners, "summary_field": key}
        return result

    non_sparseseg = [
        row for row in summary_overall if not str(row["model"]).startswith("SparseSeg")
    ]
    eligible = [
        row for row in non_sparseseg if bool(row.get("eligible_non_degenerate_summary"))
    ]
    return {
        "schema_version": 1,
        "foreground_quality_definition": (
            "absolute difference from GT foreground fraction; lower is better"
        ),
        "non_degenerate_mean_foreground_ratio_range_inclusive": [
            NON_DEGENERATE_MIN_FOREGROUND_RATIO,
            NON_DEGENERATE_MAX_FOREGROUND_RATIO,
        ],
        "all_non_sparseseg_models": build(non_sparseseg),
        "non_degenerate_non_sparseseg_models": build(eligible),
        "eligible_models": [str(row["model"]) for row in eligible],
        "excluded_models": [
            str(row["model"]) for row in non_sparseseg if row not in eligible
        ],
    }


def compatibility_metrics_dict(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        metrics[model] = {
            key: [
                [
                    float(row[key])
                    for row in sorted(
                        (item for item in model_rows if int(item["roi_num"]) == roi),
                        key=lambda item: int(item["trial"]),
                    )
                ]
                for roi in ROI_NUMS
            ]
            for key in ("absolute_iou", "relative_iou_log_normalized", "precision", "recall", "predicted_foreground_fraction")
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celltype", default="hela2")
    parser.add_argument("--organelle", default="mito")
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("--empanda-root", type=Path, default=default_empanda_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (defaults to <empanda-root>/evaluation_cross_trials_extreme).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Evaluate only this model name; repeat for multiple models.",
    )
    completeness = parser.add_mutually_exclusive_group()
    completeness.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail if any selected prediction is missing (default).",
    )
    completeness.add_argument(
        "--allow-missing",
        dest="strict",
        action="store_false",
        help="Write partial results instead of failing when predictions are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_start = time.perf_counter()
    start_utc = utc_now()
    repo = Path(__file__).resolve().parent
    gt_path = args.gt_path or repo / "inputdata" / f"{args.celltype}_{args.organelle}_s3.tif"
    output_dir = args.output_dir or args.empanda_root / "evaluation_cross_trials_extreme"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = np.squeeze(tifffile.imread(gt_path)) > 0
    if gt.ndim != 3:
        raise ValueError(f"Expected 3D GT, got {gt.shape}: {gt_path}")
    print(f"GT: {gt_path} shape={gt.shape} foreground_fraction={gt.mean():.8f}")

    specs = build_model_specs()
    if args.models:
        selected = set(args.models)
        known = {spec.name for spec in specs}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}; known={sorted(known)}")
        specs = [spec for spec in specs if spec.name in selected]
    else:
        # Experimental entries that do not satisfy the full 15-case contract
        # remain available through explicit --model selection, but must not
        # make the default strict benchmark fail.
        specs = [spec for spec in specs if spec.default_enabled]

    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for roi in ROI_NUMS:
        for trial in TRIALS:
            for spec in specs:
                prediction_path, candidates = resolve_prediction(
                    spec,
                    repo=repo,
                    empanda=args.empanda_root,
                    celltype=args.celltype,
                    organelle=args.organelle,
                    trial=trial,
                    roi=roi,
                )
                if prediction_path is None:
                    record = {
                        "model": spec.name,
                        "trial": trial,
                        "roi_num": roi,
                        "candidate_paths": " | ".join(str(path) for path in candidates),
                    }
                    missing.append(record)
                    print(f"[missing] {spec.name} trial={trial} roi={roi}")
                    continue
                prediction = spec.loader(prediction_path, **spec.loader_kwargs)
                metrics = absolute_metrics(gt, prediction)
                timing = read_timing(prediction_path)
                row: Dict[str, object] = {
                    "model": spec.name,
                    "trial": trial,
                    "roi_num": roi,
                    "prediction_path": str(prediction_path),
                    **metrics,
                    **timing,
                }
                rows.append(row)
                print(
                    f"[ok] model={spec.name} trial={trial} roi={roi} "
                    f"IoU={row['absolute_iou']:.6f} precision={row['precision']:.6f} "
                    f"recall={row['recall']:.6f} pred_fg={row['predicted_foreground_fraction']:.8f} "
                    f"e2e_s={row['end_to_end_wall_clock_seconds']}",
                    flush=True,
                )

    relative_iou_denominator = add_relative_iou(rows)

    trial_fields = [
        "model", "trial", "roi_num", "absolute_iou", "relative_iou_log_normalized", "precision", "recall",
        "predicted_foreground_fraction", "gt_foreground_fraction",
        "foreground_fraction_absolute_error", "predicted_to_gt_foreground_ratio",
        "constant_prediction", "degenerate_prediction",
        "true_positive", "false_positive", "false_negative", "true_negative",
        "predicted_foreground_voxels", "gt_foreground_voxels", "total_voxels",
        "end_to_end_wall_clock_seconds", "timing_status", "timing_path",
        "shared_preprocessing_wall_clock_seconds", "shared_preprocessing_timing_path",
        "prediction_path",
    ]
    write_csv(output_dir / "trial_metrics_absolute.csv", rows, trial_fields)
    write_csv(
        output_dir / "missing_predictions.csv",
        missing,
        ("model", "trial", "roi_num", "candidate_paths"),
    )
    summary_by_roi, summary_overall = make_summaries(rows)
    summary_fields = [
        "model", "roi_num", "completed_trials",
        "mean_absolute_iou", "std_absolute_iou",
        "mean_relative_iou_log_normalized", "std_relative_iou_log_normalized",
        "mean_precision", "std_precision", "mean_recall", "std_recall",
        "mean_predicted_foreground_fraction", "std_predicted_foreground_fraction",
        "mean_foreground_fraction_absolute_error", "std_foreground_fraction_absolute_error",
        "mean_predicted_to_gt_foreground_ratio", "std_predicted_to_gt_foreground_ratio",
        "constant_prediction_trials", "degenerate_prediction_trials",
        "eligible_non_degenerate_summary",
        "measured_timing_trials", "unique_measured_timing_runs",
        "unmeasured_timing_trials", "shared_preprocessing_runs",
        "total_end_to_end_wall_clock_seconds", "mean_end_to_end_wall_clock_seconds",
    ]
    write_csv(output_dir / "summary_by_roi.csv", summary_by_roi, summary_fields)
    write_csv(output_dir / "summary_overall.csv", summary_overall, summary_fields)
    gates = componentwise_gates(summary_overall)
    with (output_dir / "componentwise_gates.json").open("w", encoding="utf-8") as handle:
        json.dump(gates, handle, indent=2)

    saved = {
        "schema_version": 2,
        "metric_type": "absolute_voxel_metrics_plus_historical_relative_iou",
        "relative_iou_definition": "ln(1 + 100 * absolute_iou) / D",
        "relative_iou_denominator_D": relative_iou_denominator,
        "relative_iou_denominator_scope": "maximum transformed IoU over all included model/trial/ROI rows",
        "labels": list(ROI_NUMS),
        "trial_rows": rows,
        "missing_files": missing,
        "summary_by_roi": summary_by_roi,
        "summary_overall": summary_overall,
        "componentwise_gates": gates,
        "metrics_dict": compatibility_metrics_dict(rows),
    }
    with (output_dir / "metrics_cross_model_absolute.pkl").open("wb") as handle:
        pickle.dump(saved, handle)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(saved | {"metrics_dict": "see pickle"}, handle, indent=2)

    measured_model_durations = finite_values(rows, "end_to_end_wall_clock_seconds")
    unique_model_durations = unique_timing_values(
        rows, duration_key="end_to_end_wall_clock_seconds", identity_key="timing_path"
    )
    unique_shared_durations = unique_timing_values(
        rows,
        duration_key="shared_preprocessing_wall_clock_seconds",
        identity_key="shared_preprocessing_timing_path",
    )
    supplemental_iteration2_candidates = (
        args.empanda_root
        / "response"
        / "sparseseg_adaptive_iterated_v735"
        / "100_1"
        / "timing.json",
        args.empanda_root
        / "response"
        / "sparseseg_iteration2_balanced_quantile_timed_v25"
        / "100_1"
        / "timing.json",
        args.empanda_root
        / "response"
        / "sparseseg_iteration2_exact_quantile_timed_v24"
        / "100_1"
        / "timing.json",
        args.empanda_root
        / "response"
        / "sparseseg_iteration2_paperlocked_timed"
        / "100_1"
        / "timing.json",
        args.empanda_root
        / "response"
        / "sparseseg_iteration2_timed"
        / "100_1"
        / "timing.json",
    )
    supplemental_iteration2_path = next(
        (path for path in supplemental_iteration2_candidates if path.exists()),
        supplemental_iteration2_candidates[0],
    )
    supplemental_iteration2 = None
    if supplemental_iteration2_path.exists():
        with supplemental_iteration2_path.open("r", encoding="utf-8") as handle:
            supplemental_iteration2 = json.load(handle)

    paired_evaluation_wall_clock_seconds = time.perf_counter() - evaluation_start
    evaluation_timing = {
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "evaluation_wall_clock_seconds": time.perf_counter() - evaluation_start,
        "paired_evaluation_wall_clock_seconds": paired_evaluation_wall_clock_seconds,
        "evaluated_prediction_count": len(rows),
        "missing_prediction_count": len(missing),
        "measured_prediction_timing_count": len(measured_model_durations),
        "unmeasured_prediction_timing_count": len(rows) - len(measured_model_durations),
        "unique_measured_model_run_count": len(unique_model_durations),
        "unique_shared_preprocessing_run_count": len(unique_shared_durations),
        "total_measured_model_end_to_end_wall_clock_seconds": (
            sum(unique_model_durations) + sum(unique_shared_durations)
        ),
        "supplemental_sparseseg_second_iteration_timing_path": str(supplemental_iteration2_path),
        "supplemental_sparseseg_second_iteration": supplemental_iteration2,
        "note": "Totals de-duplicate shared prediction/timing paths. The supplemental SparseSeg number is one second-iteration measurement and is not a full-pipeline total.",
    }
    with (output_dir / "evaluation_timing.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation_timing, handle, indent=2)
    print(json.dumps(evaluation_timing, indent=2))
    if args.strict and missing:
        raise SystemExit(f"Strict evaluation failed: {len(missing)} predictions are missing.")


if __name__ == "__main__":
    main()
