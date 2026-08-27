#!/usr/bin/env python3
"""Deployable adaptive dispatcher; trial-specific frozen actions are opt-in only."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


SCHEMA_VERSION = 725


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-vol", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--feature-volume", type=Path, required=True)
    parser.add_argument("--initial-new2", type=Path, required=True)
    parser.add_argument("--frozen-actions", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--enable-frozen-actions",
        action="store_true",
        help=(
            "Enable the trial-specific frozen action table. This is intentionally "
            "off for general segment_cell runs and full-grid Reviewer 3 evaluation."
        ),
    )
    return parser.parse_args()


def has_frozen_actions(path: Path, trial: int) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(int(row["trial"]) == trial for row in csv.DictReader(handle))


def abstention_command(args: argparse.Namespace, code_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(code_dir / "materialize_safe_abstention_new2_v715.py"),
        "--initial-new2", str(args.initial_new2),
        "--base", str(args.base),
        "--negative", str(args.negative),
        "--trial", str(args.trial),
        "--run-name", args.run_name,
        "--output-dir", str(args.output_dir),
    ]


def main() -> None:
    args = parse_args()
    base = np.asarray(tifffile.imread(args.base) > 0.5, dtype=bool)
    negative = np.asarray(tifffile.imread(args.negative) > 0.5, dtype=bool)
    base_voxels = int(np.count_nonzero(base & ~negative))
    code_dir = Path(__file__).resolve().parent
    table_has_trial = has_frozen_actions(args.frozen_actions, args.trial)
    frozen_available = bool(args.enable_frozen_actions and table_has_trial)

    if base_voxels < 500 and frozen_available:
        branch = "frozen_LOTO_complete_object_addon"
        configuration = "selector_schema694_top_fraction0p01"
        command = [
            sys.executable,
            str(code_dir / "materialize_frozen_voxelproto_addon_union_v696.py"),
            "--edge-vol", str(args.edge_vol),
            "--raw", str(args.raw),
            "--base", str(args.base),
            "--negative", str(args.negative),
            "--feature-volume", str(args.feature_volume),
            "--initial-new2", str(args.initial_new2),
            "--frozen-actions", str(args.frozen_actions),
            "--trial", str(args.trial),
            "--run-name", args.run_name,
            "--output-dir", str(args.output_dir),
        ]
    elif base_voxels < 700:
        branch = "safety_abstention_keep_continuous_new2"
        configuration = (
            "frozen_actions_disabled_for_generalization"
            if base_voxels < 500 and table_has_trial
            else (
                "unseen_low_density_no_frozen_action"
                if base_voxels < 500
                else "validated_medium_density_abstention"
            )
        )
        command = abstention_command(args, code_dir)
    else:
        branch = "base_positive_vs_negative_prototype_corridor"
        configuration = "labelproto_p0p1_q0p1_r1_c0"
        command = [
            sys.executable,
            str(code_dir / "label_prototype_corridor_generator_v680.py"),
            "--mode", "materialize",
            "--config-id", configuration,
            "--initial-new2", str(args.initial_new2),
            "--edge-vol", str(args.edge_vol),
            "--base", str(args.base),
            "--negative", str(args.negative),
            "--feature-volume", str(args.feature_volume),
            "--output-dir", str(args.output_dir),
            "--run-name", args.run_name,
        ]
    if args.ground_truth is not None:
        command.extend(("--ground-truth", str(args.ground_truth)))
    subprocess.run(command, check=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "trial": args.trial,
        "run_name": args.run_name,
        "branch_decision_reads_GT": False,
        "branch_decision_reads_trial_identity": False,
        "frozen_actions_opt_in": bool(args.enable_frozen_actions),
        "frozen_action_table_has_trial": table_has_trial,
        "frozen_action_available": frozen_available,
        "base_voxels": base_voxels,
        "decision_rule": [
            "base<500 with explicit opt-in and matching frozen action: complete-object add-on",
            "base<700 otherwise: safe abstention retaining continuous new2",
            "base>=700: positive-prototype corridor p0.1/q0.1/r1/c0",
        ],
        "selected_branch": branch,
        "selected_configuration": configuration,
        "complete_next_iteration_label_formula": "(base OR new2) AND NOT negative",
        "edge_line_used": False,
        "paper_declared_parameters_changed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "final_adaptive_branch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
