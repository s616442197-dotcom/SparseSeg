"""Run one actual segment_cell training/prediction pass (CNN or ViT backbone)."""
from __future__ import annotations
import argparse, importlib.util, sys, time
from pathlib import Path
import random
from common import (add_standard_arguments, check_inputs, normalize_prediction,
                    stage_vem_names, write_timing)

def load_segment_cell(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_segment_cell", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("finetuning", "vit"), required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    backend = Path(__file__).resolve().parents[1] / "formal_assets" / "sparseseg_adaptive_backend"
    parser.add_argument("--adaptive-backend", type=Path, default=backend)
    parser.add_argument(
        "--continuous-selector",
        type=Path,
        default=backend / "assets" / "complete_object_grouped_oof_models.joblib",
    )
    parser.add_argument(
        "--frozen-actions",
        type=Path,
        default=backend / "assets" / "frozen_voxelproto_addon_actions_v696.csv",
    )
    args = parser.parse_args()
    raw, _, _ = check_inputs(args)
    started = time.perf_counter()
    data_root = stage_vem_names(
        args.raw,
        args.sparse_label,
        args.negative_label,
        args.work_dir / "inputdata",
        args.trial,
        args.roi_num,
    )
    candidates = [
        args.repo_root / "vemmodel" / "janelia_cosem" / "segment_cell.py",
        args.repo_root / "segment_cell.py",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"segment_cell.py not found in: {candidates}")
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(source.parent))
    module = load_segment_cell(source)
    run_dir = args.work_dir / "iterations"
    run_dir.mkdir(parents=True, exist_ok=True)
    adaptive = args.variant == "finetuning" and args.iterations > 1
    if adaptive:
        required = (args.adaptive_backend, args.continuous_selector, args.frozen_actions)
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise FileNotFoundError(f"missing adaptive backend assets: {missing}")
    for iteration in range(args.iterations):
        random.seed(args.seed + iteration)
        try:
            import numpy as np
            import torch
            np.random.seed(args.seed + iteration)
            torch.manual_seed(args.seed + iteration)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + iteration)
        except ImportError:
            pass
        module.main(
            interation_idx=iteration,
            raw_name="hela2_em_s3",
            mask_name=f"label_hela2_mito_{args.trial}_{args.roi_num}",
            folder_name=str(run_dir),
            patch_scale=80,
            inference_stride=40,
            repeated_epoch=args.epochs,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            base_folder=str(data_root),
            if_Vit=args.variant == "vit",
            Loss_list=[10.0, 0.1, 0.1, 0.05],
            refinement_profile=(
                "adaptive_iterated" if adaptive
                else "legacy" if args.variant == "vit" and args.iterations > 1
                else "safe_abstain"
            ),
            adaptive_trial=args.trial if adaptive else None,
            adaptive_run_name=f"{args.trial}_{args.roi_num}" if adaptive else None,
            adaptive_backend_dir=str(args.adaptive_backend) if adaptive else None,
            adaptive_continuous_selector=(
                str(args.continuous_selector) if adaptive else None
            ),
            adaptive_frozen_actions=str(args.frozen_actions) if adaptive else None,
        )
    prediction = run_dir / "prediction_fixed_threshold.tif"
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    normalize_prediction(prediction, args.output, raw.shape)
    write_timing(
        args.output,
        model=f"sparseseg_{args.variant}",
        started=started,
        epochs=args.epochs,
        extra={
            "iterations": args.iterations,
            "seed": args.seed,
            "refinement_profile": (
                "adaptive_iterated" if adaptive
                else "legacy" if args.variant == "vit" and args.iterations > 1
                else "safe_abstain"
            ),
        },
    )

if __name__ == "__main__":
    main()
