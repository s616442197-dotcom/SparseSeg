"""Run the 12 real model adapters on the packaged raw example, evaluate, then plot.

The formal reviewer figures are regenerated from data/*.csv.  The one-epoch
example predictions are evaluated separately in outputs/example_evaluation_metrics.csv.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline_config.example.json"
RAW = ROOT / "example_data" / "raw.tif"
GT = ROOT / "example_data" / "gt.tif"
SPARSE = ROOT / "example_data" / "sparse_label.tif"
NEGATIVE = ROOT / "example_data" / "negative_label.tif"
PREDICTIONS = ROOT / "example_data" / "predictions"
WORK = ROOT / "example_data" / "work"
OUTPUTS = ROOT / "outputs"

def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

def expand(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "${" in expanded or ("%" in expanded and expanded.count("%") >= 2):
        raise RuntimeError(f"Unresolved environment variable: {value}")
    return expanded

def load_config(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 12:
        raise ValueError("pipeline config must contain exactly 12 models")
    names = [str(item.get("name")) for item in models]
    if len(set(names)) != 12:
        raise ValueError("model names must be unique")
    return models

def validate_inputs() -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    import tifffile
    required = (RAW, GT, SPARSE, NEGATIVE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing packaged example inputs: {missing}")
    raw, gt, sparse, negative = (np.asarray(tifffile.imread(p)) for p in required)
    if len({a.shape for a in (raw, gt, sparse, negative)}) != 1:
        raise ValueError("example raw/GT/sparse/negative TIFF shapes do not match")
    if not np.any(gt > 0) or not np.any(sparse > 0):
        raise ValueError("GT and sparse label must each contain foreground")
    return raw, gt > 0

def build_command(item: dict[str, object], output: Path, work_dir: Path, epochs: int) -> list[str]:
    python = expand(str(item["python"]))
    adapter = ROOT / str(item["adapter"])
    if not Path(python).is_file():
        raise FileNotFoundError(f"Python interpreter not found for {item['name']}: {python}")
    if not adapter.is_file():
        raise FileNotFoundError(f"Adapter not found for {item['name']}: {adapter}")
    extra = [expand(str(value)) for value in item.get("args", [])]
    return [python, str(adapter), *extra,
            "--raw", str(RAW), "--sparse-label", str(SPARSE),
            "--negative-label", str(NEGATIVE), "--output", str(output),
            "--work-dir", str(work_dir), "--epochs", str(epochs),
            "--device", str(item.get("device", "cuda"))]

def compute_metrics(model: str, prediction_path: Path, gt: np.ndarray, elapsed: float,
                    epochs: int) -> dict[str, object]:
    import numpy as np
    import tifffile
    pred = np.squeeze(np.asarray(tifffile.imread(prediction_path))) > 0
    if pred.shape != gt.shape:
        raise ValueError(f"{model}: prediction shape {pred.shape} != GT shape {gt.shape}")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    union = tp + fp + fn
    return {
        "model": model,
        "epochs": epochs,
        "absolute_iou": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_foreground_fraction": float(pred.mean()),
        "gt_foreground_fraction": float(gt.mean()),
        "wall_clock_seconds": elapsed,
        "prediction": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
    }

def write_metrics(rows: list[dict[str, object]]) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    target = OUTPUTS / "example_evaluation_metrics.csv"
    fields = list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return target

def run_models(models: list[dict[str, object]], selected: set[str] | None,
               epochs: int, skip_existing: bool) -> Path:
    _, gt = validate_inputs()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for item in models:
        name = str(item["name"])
        if selected and name not in selected:
            continue
        output = PREDICTIONS / f"{slug(name)}.tif"
        work_dir = WORK / slug(name)
        started = time.perf_counter()
        timing_path = output.with_suffix(".timing.json")
        elapsed = None
        effective_epochs = epochs
        if output.exists() and skip_existing:
            print(f"[reuse] {name}: {output}")
        else:
            command = build_command(item, output, work_dir, epochs)
            print(f"\n=== {name} ===", flush=True)
            print("[command]", subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            elapsed = float(timing["wall_clock_seconds"])
            effective_epochs = int(timing.get("epochs", epochs))
        if elapsed is None:
            elapsed = time.perf_counter() - started
        if not output.is_file():
            raise FileNotFoundError(f"{name} did not create required output: {output}")
        rows.append(compute_metrics(name, output, gt, elapsed, effective_epochs))
    if selected and len(rows) != len(selected):
        found = {str(row["model"]) for row in rows}
        raise ValueError(f"Unknown --models entries: {sorted(selected - found)}")
    return write_metrics(rows)

def regenerate_figures(
    metrics_csv: Path,
    time_csv: Path,
    wallclock_csv: Path,
    output_dir: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_benchmark_figures.py"),
            "--metrics",
            str(metrics_csv),
            "--time",
            str(time_csv),
            "--wallclock",
            str(wallclock_csv),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=ROOT,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=1,
                        help="One-epoch compatibility setting; formal CSVs remain unchanged")
    parser.add_argument("--models", nargs="*", help="Optional exact model-name subset")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--metrics-csv", type=Path, default=ROOT / "data" / "benchmark_metrics.csv")
    parser.add_argument("--time-csv", type=Path, default=ROOT / "data" / "time_per_epoch.csv")
    parser.add_argument("--wallclock-csv", type=Path, default=ROOT / "data" / "wallclock_total.csv")
    parser.add_argument("--figure-output-dir", type=Path, default=OUTPUTS)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    models = load_config(args.config.resolve())
    if not args.plots_only:
        metrics = run_models(models, set(args.models) if args.models else None,
                             args.epochs, args.skip_existing)
        print(f"\nExample metrics: {metrics}")
    if not args.no_plots:
        figure_output_dir = args.figure_output_dir.resolve()
        regenerate_figures(
            args.metrics_csv.resolve(),
            args.time_csv.resolve(),
            args.wallclock_csv.resolve(),
            figure_output_dir,
        )
        print(f"Figures: {figure_output_dir / 'Fig_add1_v3.pdf'}")
        print(f"         {figure_output_dir / 'Fig_add_more1_v3.pdf'}")

if __name__ == "__main__":
    main()
