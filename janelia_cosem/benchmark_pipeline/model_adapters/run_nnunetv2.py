"""Run official nnU-Net v2 raw or VEM sparse-matched trainer for one example."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
from common import (add_standard_arguments, check_inputs, normalize_prediction, run,
                    stage_vem_names, write_timing)

def executable(name: str) -> Path:
    path = Path(sys.executable).parent / name
    if os.name == "nt" and not path.exists():
        path = path.with_suffix(".exe")
    if not path.exists():
        raise FileNotFoundError(f"nnU-Net executable missing in environment: {path}")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="Folder containing prepare_official_nnunetv2_data.py")
    parser.add_argument("--variant", choices=("raw", "sparse_matched"), required=True)
    args = parser.parse_args()
    if args.epochs not in (1, 50):
        raise ValueError("nnU-Net adapter supports the archived 1- or 50-epoch trainers")
    raw, _, _ = check_inputs(args)
    started = time.perf_counter()
    source_root = stage_vem_names(args.raw, args.sparse_label, args.negative_label,
                                  args.work_dir / "inputdata", args.trial, args.roi_num)
    pipeline = args.work_dir / "nnunet_pipeline"
    nn_raw = pipeline / "nnUNet_raw"
    nn_preprocessed = pipeline / "nnUNet_preprocessed"
    nn_results = pipeline / "nnUNet_results"
    for path in (nn_raw, nn_preprocessed, nn_results):
        path.mkdir(parents=True, exist_ok=True)
    prepare = args.code_root / "prepare_official_nnunetv2_data.py"
    if not prepare.is_file():
        raise FileNotFoundError(prepare)
    roi_offset = {1: 0, 5: 5, 10: 10}[args.roi_num]
    task_id = roi_offset + (args.trial - 100)
    if not 0 <= task_id < 15:
        raise ValueError(f"unsupported formal case: trial={args.trial}, roi={args.roi_num}")
    dataset_id = (701 if args.variant == "raw" else 751) + task_id
    suffix = "Raw" if args.variant == "raw" else "SparseMatched"
    dataset_name = (
        f"Dataset{dataset_id:03d}_VEMHela2MitoT{args.trial}R{args.roi_num}{suffix}"
    )
    if args.variant == "raw":
        trainer = "nnUNetTrainer_1epoch" if args.epochs == 1 else "nnUNetTrainer_50epochs"
    else:
        trainer = (
            "nnUNetTrainerVEMSparseMatched1epoch"
            if args.epochs == 1
            else "nnUNetTrainerVEMSparseMatched50epochs"
        )
    env = dict(os.environ)
    env.update({"nnUNet_raw": str(nn_raw), "nnUNet_preprocessed": str(nn_preprocessed),
                "nnUNet_results": str(nn_results),
                "nnUNet_extTrainer": str(args.code_root / "nnunet_ext_trainers"),
                "nnUNet_n_proc_DA": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    run([sys.executable, prepare, "prepare-common", "--source-root", source_root,
         "--work-root", pipeline], env=env)
    run([sys.executable, prepare, "prepare-dataset", "--variant", args.variant,
         "--task-id", task_id, "--source-root", source_root, "--work-root", pipeline,
         "--nnunet-raw", nn_raw], env=env)
    run([sys.executable, prepare, "validate-dataset", "--variant", args.variant,
         "--task-id", task_id, "--nnunet-raw", nn_raw], env=env)
    plan_command = [executable("nnUNetv2_plan_and_preprocess"), "-d", dataset_id,
                    "-c", "2d", "-npfp", "2", "-np", "2", "--no_pbar"]
    if args.variant == "raw":
        plan_command.append("--verify_dataset_integrity")
    run(plan_command, env=env)
    run([executable("nnUNetv2_train"), dataset_id, "2d", "all", "-tr", trainer,
         "-device", args.device], env=env)
    prediction_dir = args.work_dir / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    run([executable("nnUNetv2_predict"), "-i", nn_raw / dataset_name / "imagesTs",
         "-o", prediction_dir, "-d", dataset_id, "-c", "2d", "-f", "all",
         "-tr", trainer, "-chk", "checkpoint_final.pth", "-device", args.device,
         "-npp", "1", "-nps", "1", "--disable_progress_bar"], env=env)
    prediction = prediction_dir / f"hela2_t{args.trial}_r{args.roi_num}.tif"
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    normalize_prediction(prediction, args.output, raw.shape)
    write_timing(args.output, model=f"nnunetv2_{args.variant}", started=started,
                 epochs=args.epochs, extra={"trainer": trainer})

if __name__ == "__main__":
    main()
