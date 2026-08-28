"""Run the actual Vanilla U-Net raw or sparse-matched benchmark implementation."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from common import (add_standard_arguments, check_inputs, find_one, normalize_prediction,
                    run, stage_vem_names, write_timing)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--variant", choices=("vanilla_unet", "vanilla_unet_sparse_matched"),
                        required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    raw, _, _ = check_inputs(args)
    started = time.perf_counter()
    data_root = stage_vem_names(args.raw, args.sparse_label, args.negative_label,
                                args.work_dir / "inputdata", args.trial, args.roi_num)
    script = (
        Path(__file__).resolve().parents[1]
        / "formal_runners"
        / "vanilla_unet"
        / "sparse_baseline_benchmark.py"
    )
    if not script.is_file():
        raise FileNotFoundError(script)
    result_root = args.work_dir / "runner_output"
    command = [sys.executable, script, "--variant", args.variant,
               "--trial", args.trial, "--roi-num", args.roi_num, "--data-root", data_root,
               "--output-root", result_root, "--epochs", args.epochs,
               "--seed", args.seed,
               "--num-samples", args.num_samples, "--batch-size", args.batch_size,
               "--patch-size", "80", "--inference-tile-size", "256",
               "--inference-overlap", "32", "--inference-batch-size", "2",
               "--num-workers", "0", "--overwrite"]
    if args.device == "cpu":
        command.append("--no-amp")
    run(command)
    source = find_one(
        result_root,
        [f"**/prediction_hela2_mito_{args.trial}_{args.roi_num}.tif"],
    )
    normalize_prediction(source, args.output, raw.shape)
    write_timing(args.output, model=args.variant, started=started, epochs=args.epochs)

if __name__ == "__main__":
    main()
