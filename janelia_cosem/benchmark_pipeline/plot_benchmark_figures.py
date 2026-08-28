#!/usr/bin/env python3
"""Generate the two packaged 12-model benchmark figures."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
FONT_SIZE = 12
ROI_NUMS = (1, 5, 10)
MODELS = (
    "SparseSeg",
    "SparseSeg-ViT",
    "Vanilla-UNet",
    "Vanilla-UNet-SparseMatched",
    "nnU-Net",
    "nnU-Net-SparseMatched",
    "MitoNet-Pretrained",
    "MitoNet-Sparse-Finetuned",
    "StarDist",
    "DeePict",
    "COSEM-2D-UNet",
    "COSEM-3D-UNet",
)
FIG_ADD1_MODELS = tuple(
    model
    for model in MODELS
    if model not in ("Vanilla-UNet-SparseMatched", "nnU-Net-SparseMatched")
)
DISPLAY = {
    "SparseSeg": "SparseSeg",
    "SparseSeg-ViT": "SparseSeg-ViT",
    "Vanilla-UNet": "Vanilla U-Net raw",
    "Vanilla-UNet-SparseMatched": "Vanilla U-Net + sparse-matched",
    "nnU-Net": "nnU-Net raw",
    "nnU-Net-SparseMatched": "nnU-Net + sparse-matched",
    "MitoNet-Pretrained": "MitoNet pretrained",
    "MitoNet-Sparse-Finetuned": "MitoNet sparse-adapted",
    "StarDist": "StarDist",
    "DeePict": "DeePiCt",
    "COSEM-2D-UNet": "COSEM 2D U-Net",
    "COSEM-3D-UNet": "COSEM 3D U-Net",
}
COLORS = {model: plt.get_cmap("tab20")(index) for index, model in enumerate(MODELS)}
METRICS = (
    ("relative_iou", "Relative IoU"),
    ("absolute_iou", "Absolute IoU"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("predicted_foreground_fraction", "Predicted foreground fraction"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_metrics(path: Path) -> tuple[list[dict], float]:
    required = {
        "model",
        "trial",
        "roi_num",
        "absolute_iou",
        "precision",
        "recall",
        "predicted_foreground_fraction",
        "gt_foreground_fraction",
    }
    rows: list[dict] = []
    for raw in read_csv(path):
        if not required.issubset(raw):
            raise ValueError(f"Metric CSV is missing columns: {sorted(required - set(raw))}")
        if raw["model"] not in MODELS:
            continue
        row = dict(raw)
        row["trial"] = int(row["trial"])
        row["roi_num"] = int(row["roi_num"])
        for field in (
            "absolute_iou",
            "precision",
            "recall",
            "predicted_foreground_fraction",
            "gt_foreground_fraction",
        ):
            row[field] = float(row[field])
        rows.append(row)

    counts = Counter(row["model"] for row in rows)
    bad = {model: counts[model] for model in MODELS if counts[model] != 15}
    if bad:
        raise ValueError(f"Expected 15 trial/ROI rows per model; got {bad}")
    expected_cases = {(trial, roi) for trial in range(100, 105) for roi in ROI_NUMS}
    for model in MODELS:
        cases = {
            (row["trial"], row["roi_num"])
            for row in rows
            if row["model"] == model
        }
        if cases != expected_cases:
            raise ValueError(f"Incomplete trial/ROI grid for {model}: {sorted(cases)}")

    transformed = [math.log1p(100.0 * row["absolute_iou"]) for row in rows]
    denominator = max(transformed)
    if denominator <= 0:
        raise ValueError("Relative-IoU denominator must be positive")
    for row, value in zip(rows, transformed):
        row["relative_iou"] = value / denominator

    gt_values = np.asarray(
        [row["gt_foreground_fraction"] for row in rows], dtype=float
    )
    if not np.allclose(gt_values, gt_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Expected one common GT foreground fraction")
    return rows, float(gt_values[0])


def read_time_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in read_csv(path):
        if raw["model"] not in MODELS:
            continue
        row = dict(raw)
        row["trial"] = int(row["trial"])
        row["roi_num"] = int(row["roi_num"])
        row["time_minutes"] = float(row["time_minutes"])
        if row["time_minutes"] <= 0:
            raise ValueError(f"Non-positive time/epoch row: {raw}")
        rows.append(row)
    counts = Counter(row["model"] for row in rows)
    missing = [model for model in MODELS if counts[model] == 0]
    if missing:
        raise ValueError(f"Missing time/epoch observations: {missing}")
    return rows


def read_wallclock(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    seconds: dict[str, float] = {}
    for row in rows:
        model = row["entry"]
        if model in MODELS:
            value = float(row["mean_wall_clock_seconds"])
            if value <= 0:
                raise ValueError(f"Non-positive wall-clock value for {model}")
            if model in seconds:
                raise ValueError(f"Duplicate wall-clock row for {model}")
            seconds[model] = value
    missing = [model for model in MODELS if model not in seconds]
    if missing:
        raise ValueError(f"Missing wall-clock values: {missing}")
    return seconds


def group_metrics(rows: list[dict]) -> dict:
    grouped = {
        model: {metric: defaultdict(list) for metric, _ in METRICS}
        for model in MODELS
    }
    for row in rows:
        for metric, _ in METRICS:
            grouped[row["model"]][metric][row["roi_num"]].append(row[metric])
    return {
        model: {
            metric: [grouped[model][metric][roi] for roi in ROI_NUMS]
            for metric, _ in METRICS
        }
        for model in MODELS
    }


def lighten(color: tuple[float, ...], amount: float = 0.32) -> tuple[float, ...]:
    rgba = np.asarray(mcolors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * amount
    rgba[3] = 0.92
    return tuple(rgba)


def plot_metric(
    ax: plt.Axes,
    grouped: dict,
    metric: str,
    label: str,
    panel: str,
    models: tuple[str, ...] = MODELS,
    gt_foreground_fraction: float | None = None,
) -> None:
    centers = np.arange(3, dtype=float)
    width = 0.064
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    for model_index, model in enumerate(models):
        positions = centers + offsets[model_index]
        border = lighten(COLORS[model])
        for roi_index, current in enumerate(grouped[model][metric]):
            values = np.asarray(current, dtype=float)
            density_values = values.copy()
            if np.ptp(density_values) <= 1e-12:
                center = float(density_values[0])
                epsilon = (
                    max(4e-6, min(5e-3, abs(center) * 0.015))
                    if metric == "predicted_foreground_fraction"
                    else 0.003
                )
                density_values = center + np.linspace(-epsilon, epsilon, len(values))
            violin = ax.violinplot(
                [density_values],
                positions=[positions[roi_index]],
                widths=width * 0.92,
                showmeans=False,
                showmedians=True,
                showextrema=True,
                bw_method=0.6,
                points=80,
            )
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[model])
                body.set_edgecolor(border)
                body.set_linewidth(0.65)
                body.set_alpha(0.48)
            for key in ("cmins", "cmaxes", "cbars"):
                violin[key].set_color(border)
                violin[key].set_linewidth(0.55)
            violin["cmedians"].set_color("#555555")
            violin["cmedians"].set_linewidth(0.75)
            jitter = np.linspace(-width * 0.20, width * 0.20, len(current))
            ax.scatter(
                np.full(len(current), positions[roi_index]) + jitter,
                current,
                s=7,
                facecolor="white",
                edgecolor="#666666",
                linewidth=0.32,
                zorder=4,
            )
    ax.set_xlim(-0.48, 2.48)
    ax.set_xticks(centers, [str(value) for value in ROI_NUMS])
    ax.set_xlabel("Number of positive ROIs", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_ylabel(label, fontsize=FONT_SIZE, fontweight="bold")
    ax.set_title(f"{panel}  {label}", loc="left", fontsize=FONT_SIZE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    if metric == "predicted_foreground_fraction":
        if gt_foreground_fraction is None:
            raise ValueError("GT foreground fraction is required")
        ax.set_yscale("symlog", linthresh=1e-4, linscale=0.8)
        ax.set_ylim(-1e-5, 1.2)
        ax.axhline(
            gt_foreground_fraction,
            color="#444444",
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        ax.annotate(
            f"GT = {gt_foreground_fraction:.5f}",
            xy=(2.46, gt_foreground_fraction),
            xytext=(-2, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE,
            color="#333333",
        )
    else:
        ax.set_ylim(-0.015, 1.045)


def plot_time(
    ax: plt.Axes,
    metrics: list[dict],
    time_rows: list[dict],
    models: tuple[str, ...],
) -> None:
    relative = {
        (row["model"], row["trial"], row["roi_num"]): row["relative_iou"]
        for row in metrics
    }
    displayed_times: list[float] = []
    for model in models:
        current = [row for row in time_rows if row["model"] == model]
        x = np.asarray([row["time_minutes"] for row in current], dtype=float)
        y = np.asarray(
            [relative[(model, row["trial"], row["roi_num"])] for row in current],
            dtype=float,
        )
        displayed_times.extend(x.tolist())
        inference_only = all(
            row["measurement_kind"] == "inference_time_per_volume_no_training_epoch"
            for row in current
        )
        ax.scatter(
            x,
            y,
            s=31 if inference_only else 24,
            marker="X" if inference_only else "o",
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.86,
            zorder=3,
        )
    all_times = np.asarray(displayed_times, dtype=float)
    ax.set_xscale("log")
    ax.set_xlim(
        10 ** (math.log10(all_times.min()) - 0.14),
        10 ** (math.log10(all_times.max()) + 0.14),
    )
    ax.set_ylim(-0.015, 1.045)
    ax.set_xlabel("Average training time per epoch (min)", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_ylabel("Relative IoU", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_title("b  Average time/epoch", loc="left", fontsize=FONT_SIZE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", color="#D6D6D6", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def plot_wallclock(ax: plt.Axes, seconds: dict[str, float]) -> None:
    positions = np.arange(len(MODELS), dtype=float)
    minutes = np.asarray([seconds[model] / 60.0 for model in MODELS], dtype=float)
    for index, model in enumerate(MODELS):
        ax.scatter(
            index,
            minutes[index],
            s=52,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_ylim(minutes.min() * 0.55, minutes.max() * 1.9)
    ax.set_xticks(positions, [DISPLAY[model] for model in MODELS], rotation=52, ha="right")
    ax.set_ylabel("End-to-end wall-clock (min, log scale)", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Model", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_title("f  End-to-end wall-clock time", loc="left", fontsize=FONT_SIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", color="#D6D6D6", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def legend_handles(models: tuple[str, ...]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=DISPLAY[model],
        )
        for model in models
    ]


def save_figure(fig: plt.Figure, pdf_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build(
    metrics_path: Path,
    time_path: Path,
    wallclock_path: Path,
    output_dir: Path,
    save_png: bool = False,
) -> tuple[Path, Path]:
    metrics, gt_foreground_fraction = read_metrics(metrics_path)
    time_rows = read_time_rows(time_path)
    wallclock = read_wallclock(wallclock_path)
    grouped = group_metrics(metrics)
    time_rows_add1 = [row for row in time_rows if row["model"] in FIG_ADD1_MODELS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_SIZE})
    fig_add1, axes_add1 = plt.subplots(1, 2, figsize=(16.8, 6.8))
    plot_metric(
        axes_add1[0],
        grouped,
        "relative_iou",
        "Relative IoU",
        "a",
        models=FIG_ADD1_MODELS,
    )
    mitonet_reference = float(
        np.median(
            [
                row["relative_iou"]
                for row in metrics
                if row["model"] == "MitoNet-Pretrained"
            ]
        )
    )
    axes_add1[0].axhline(
        mitonet_reference,
        color=COLORS["MitoNet-Pretrained"],
        linestyle=(0, (5, 3)),
        linewidth=1.35,
        alpha=0.9,
        zorder=2,
    )
    axes_add1[0].annotate(
        "MitoNet pretrained median",
        xy=(2.45, mitonet_reference),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE,
        color=mcolors.to_hex(COLORS["MitoNet-Pretrained"]),
    )
    plot_time(axes_add1[1], metrics, time_rows_add1, FIG_ADD1_MODELS)
    fig_add1.legend(
        handles=legend_handles(FIG_ADD1_MODELS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig_add1.subplots_adjust(
        left=0.075, right=0.992, top=0.94, bottom=0.23, wspace=0.27
    )
    add1_pdf = output_dir / "Fig_add1_v3.pdf"
    save_figure(
        fig_add1,
        add1_pdf,
        output_dir / "Fig_add1_v3.png" if save_png else None,
    )

    fig_more, axes_more = plt.subplots(3, 2, figsize=(16.8, 17.0))
    for axis, (metric, label), panel in zip(
        axes_more.flat[:5], METRICS, ("a", "b", "c", "d", "e")
    ):
        plot_metric(
            axis,
            grouped,
            metric,
            label,
            panel,
            gt_foreground_fraction=gt_foreground_fraction,
        )
    axes_more[0, 1].set_ylim(0.0, 0.4)
    plot_wallclock(axes_more[2, 1], wallclock)
    fig_more.legend(
        handles=legend_handles(MODELS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig_more.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.97,
        bottom=0.23,
        hspace=0.5,
        wspace=0.27,
    )
    more_pdf = output_dir / "Fig_add_more1_v3.pdf"
    save_figure(
        fig_more,
        more_pdf,
        output_dir / "Fig_add_more1_v3.png" if save_png else None,
    )
    return add1_pdf, more_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "data" / "benchmark_metrics.csv")
    parser.add_argument("--time", type=Path, default=HERE / "data" / "time_per_epoch.csv")
    parser.add_argument("--wallclock", type=Path, default=HERE / "data" / "wallclock_total.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--png", action="store_true", help="Also save PNG previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build(
        args.metrics, args.time, args.wallclock, args.output_dir, save_png=args.png
    )
    for path in outputs:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
