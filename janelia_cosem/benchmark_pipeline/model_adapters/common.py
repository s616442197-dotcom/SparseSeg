"""Shared utilities for the independently runnable 12-model example."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import tifffile

def add_standard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sparse-label", type=Path, required=True)
    parser.add_argument("--negative-label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--trial", type=int, default=100)
    parser.add_argument("--roi-num", type=int, choices=(1, 5, 10), default=1)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--device", default="cuda")

def check_inputs(args: argparse.Namespace):
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    arrays = tuple(tifffile.imread(p) for p in (args.raw, args.sparse_label, args.negative_label))
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"Input shapes differ: {[a.shape for a in arrays]}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return arrays

def stage_vem_names(raw: Path, sparse: Path, negative: Path, target: Path,
                    trial: int = 100, roi_num: int = 1) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for source, name in ((raw, "hela2_em_s3.tif"),
                         (sparse, f"label_hela2_mito_{trial}_{roi_num}.tif"),
                         (negative, "negative_hela2_em_s3.tif")):
        shutil.copy2(source, target / name)
    return target

def run(command: Sequence[object], *, cwd: Path | None = None,
        env: dict[str, str] | None = None) -> None:
    command = [str(x) for x in command]
    print("[run]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)

def find_one(root: Path, patterns: Iterable[str]) -> Path:
    patterns = list(patterns)
    matches = sorted({p.resolve() for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    if not matches:
        raise FileNotFoundError(f"No prediction below {root}; patterns={patterns}")
    matches.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)))
    if len(matches) > 1:
        print(f"[warning] using newest of {len(matches)} matches: {matches[-1]}")
    return matches[-1]

def normalize_prediction(source: Path, output: Path, reference_shape: tuple[int, ...],
                         threshold: float = 0.5) -> None:
    pred = np.squeeze(np.asarray(tifffile.imread(source)))
    if pred.shape != reference_shape:
        raise ValueError(f"Prediction shape mismatch: expected {reference_shape}, got {pred.shape}")
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    binary = pred > 0 if np.issubdtype(pred.dtype, np.integer) else pred >= threshold
    tifffile.imwrite(output, binary.astype(np.uint8), compression="zlib")

def write_timing(output: Path, *, model: str, started: float, epochs: int,
                 extra: dict[str, object] | None = None) -> None:
    payload = {"model": model, "epochs": int(epochs),
               "wall_clock_seconds": float(time.perf_counter() - started),
               "prediction": str(output.resolve())}
    if extra:
        payload.update(extra)
    output.with_suffix(".timing.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def prepend_pythonpath(env: dict[str, str], *paths: Path) -> dict[str, str]:
    result = dict(env)
    values = [str(p) for p in paths if p]
    if result.get("PYTHONPATH"):
        values.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(values)
    return result

def copy_if_different(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        shutil.copy2(source, destination)

def script_main_error_boundary(main_function) -> None:
    try:
        main_function()
    except subprocess.CalledProcessError as error:
        print(f"[error] subprocess exited with code {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode) from error
