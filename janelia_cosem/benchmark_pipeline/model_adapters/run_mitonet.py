"""Run official MitoNet pretrained inference or sparse adaptation."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
from common import (add_standard_arguments, check_inputs, find_one, normalize_prediction,
                    run, stage_vem_names, write_timing)

ASSET_MANIFEST = Path(__file__).resolve().parents[1] / "formal_assets" / "external_assets.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--variant", choices=("pretrained", "sparse_adapted"), required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--finetune-template", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    raw, _, _ = check_inputs(args)
    assets = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))["mitonet"]
    config_hash = require_sha256(
        args.base_config, assets["base_config"]["sha256"], "MitoNet base YAML"
    )
    checkpoint_hash = require_sha256(
        args.base_model, assets["checkpoint"]["sha256"], "MitoNet checkpoint"
    )
    started = time.perf_counter()
    code = Path(__file__).resolve().parents[1] / "formal_runners" / "mitonet"
    data = stage_vem_names(args.raw, args.sparse_label, args.negative_label,
                           args.work_dir / "inputdata", args.trial, args.roi_num)
    result = args.work_dir / "runner_output"
    if args.variant == "pretrained":
        script = code / "mitonet_pretrained_benchmark.py"
        command = [sys.executable, script, "--data-root", data, "--output-root", result,
                   "--base-config", args.base_config, "--base-model", args.base_model,
                   "--raw-name", "hela2_em_s3.tif", "--force"]
        patterns = ["mitonet_pretrained/masks/raw.tiff"]
    else:
        if not args.finetune_template:
            raise ValueError("--finetune-template is required for sparse_adapted")
        script = code / "mitonet_benchmark.py"
        steps_per_epoch = max(1, 200 // args.batch_size)
        command = [sys.executable, script, "--trial", args.trial, "--roi-num", args.roi_num,
                   "--data-root", data, "--output-root", result,
                   "--work-root", args.work_dir / "empanada_work",
                   "--base-config", args.base_config,
                   "--finetune-template", args.finetune_template,
                   "--base-model", args.base_model,
                   "--iterations", args.epochs * steps_per_epoch,
                   "--seed", args.seed,
                   "--batch-size", args.batch_size, "--workers", "0",
                   "--patch-size", "256", "--force"]
        patterns = [f"mitonet_benchmark/{args.trial}_{args.roi_num}/masks/raw.tiff"]
    if not script.is_file():
        raise FileNotFoundError(script)
    run(command)
    source = find_one(result, patterns)
    normalize_prediction(source, args.output, raw.shape)
    write_timing(args.output, model=f"mitonet_{args.variant}", started=started,
                 epochs=0 if args.variant == "pretrained" else args.epochs,
                 extra={"display_name": "MitoNet pretrained" if args.variant == "pretrained"
                        else "MitoNet sparse-adapted",
                        "base_config_sha256": config_hash,
                        "checkpoint_sha256": checkpoint_hash})

if __name__ == "__main__":
    main()
