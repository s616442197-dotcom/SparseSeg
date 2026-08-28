#!/usr/bin/env python3
"""Bridge SparseSeg inference to the validated adaptive new2 backend.

The backend is intentionally supplied as a directory so the released training
code contains no fitted development selector or benchmark output. The bridge
uses only runtime-visible inputs, writes an auditable manifest, and returns the
exact disjoint ``new2`` and complete next-iteration label.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tifffile


BRIDGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AdaptiveMaskResult:
    new2: np.ndarray
    complete_label: np.ndarray
    new2_path: Path
    complete_label_path: Path
    previous_base_path: Path
    bridge_manifest_path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_path(value: str | Path | None, env_name: str, description: str) -> Path:
    raw = value if value is not None else os.environ.get(env_name)
    if not raw:
        raise RuntimeError(
            f"{description} is required; pass it explicitly or set {env_name}"
        )
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def infer_case(mask_name: str, trial: int | None, run_name: str | None) -> tuple[int, str]:
    if trial is not None and run_name:
        return int(trial), str(run_name)
    match = re.search(r"(?P<trial>\d{3})_(?P<roi>1|5|10)$", mask_name)
    if match is None:
        raise ValueError(
            "adaptive trial/run name cannot be inferred from mask_name; "
            "pass adaptive_trial and adaptive_run_name"
        )
    inferred_trial = int(match.group("trial"))
    inferred_run = f"{inferred_trial}_{int(match.group('roi'))}"
    return int(trial) if trial is not None else inferred_trial, run_name or inferred_run


def _summary_path(final_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in final_dir.glob("*_summary.json")
        if path.name != "final_adaptive_branch_summary.json"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one leaf materialization summary in {final_dir}, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def generate_adaptive_iterated_mask(
    *,
    edge_vol: np.ndarray,
    raw_path: str | Path,
    feature_volume_path: str | Path,
    base_label: np.ndarray,
    negative_label: np.ndarray,
    output_folder: str | Path,
    iteration_index: int,
    trial: int,
    run_name: str,
    backend_dir: str | Path | None = None,
    continuous_selector: str | Path | None = None,
    frozen_actions: str | Path | None = None,
    probability_threshold: float = 0.27,
    top_fraction: float = 0.10,
) -> AdaptiveMaskResult:
    backend = _required_path(
        backend_dir, "SPARSESEG_ADAPTIVE_BACKEND", "adaptive backend directory"
    )
    selector = _required_path(
        continuous_selector,
        "SPARSESEG_CONTINUOUS_SELECTOR",
        "continuous complete-object selector",
    )
    actions = _required_path(
        frozen_actions,
        "SPARSESEG_FROZEN_ACTIONS",
        "frozen low-density actions CSV",
    )
    raw = Path(raw_path).resolve()
    feature = Path(feature_volume_path).resolve()
    for required in (raw, feature):
        if not required.exists():
            raise FileNotFoundError(required)
    continuous_entry = backend / "materialize_oof_continuous_policy_new2_v800.py"
    if not continuous_entry.is_file():
        continuous_entry = backend / "materialize_oof_continuous_policy_new2_v735.py"
    final_entry = backend / "materialize_final_adaptive_iterated_mask_v725.py"
    for required in (continuous_entry, final_entry):
        if not required.is_file():
            raise FileNotFoundError(required)

    probability = np.asarray(edge_vol, dtype=np.float32)
    base = np.asarray(base_label > 0.5, dtype=bool)
    negative = np.asarray(negative_label > 0.5, dtype=bool)
    if probability.shape != base.shape or negative.shape != base.shape:
        raise ValueError("edge_vol, base and negative shapes differ")
    if not np.all(np.isfinite(probability)):
        raise ValueError("edge_vol contains non-finite values")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("edge_vol must lie in [0, 1]")
    base &= ~negative

    root = Path(output_folder).resolve() / "adaptive_iterated_mask" / f"iteration_{iteration_index}"
    manifest_path = root / "adaptive_iterated_mask_bridge_manifest.json"
    final_dir = root / "final"
    if manifest_path.is_file():
        new2_path = final_dir / "test_volume_label_new2.tif"
        complete_path = final_dir / "test_volume_label_save.tif"
        previous_base_path = root / "input" / "base_input.tif"
        previous_edge_path = root / "input" / "edge_vol_probability_float32.tif"
        previous_negative_path = root / "input" / "negative_input.tif"
        for required in (
            new2_path, complete_path, previous_base_path,
            previous_edge_path, previous_negative_path,
        ):
            if not required.is_file():
                raise RuntimeError(f"incomplete resumed adaptive output: {required}")
        stored_probability = np.asarray(
            tifffile.imread(previous_edge_path), dtype=np.float32
        )
        stored_base = np.asarray(tifffile.imread(previous_base_path) > 0, dtype=bool)
        stored_negative = np.asarray(
            tifffile.imread(previous_negative_path) > 0, dtype=bool
        )
        if not np.array_equal(stored_probability, probability):
            raise RuntimeError("resumed adaptive edge_vol differs from current input")
        if not np.array_equal(stored_base, base):
            raise RuntimeError("resumed adaptive base differs from current input")
        if not np.array_equal(stored_negative, negative):
            raise RuntimeError("resumed adaptive negative label differs from current input")
        new2_bool = np.asarray(tifffile.imread(new2_path) > 0, dtype=bool)
        complete_bool = np.asarray(tifffile.imread(complete_path) > 0, dtype=bool)
        if np.any(new2_bool & stored_base) or np.any(new2_bool & stored_negative):
            raise RuntimeError("resumed adaptive new2 violates disjointness")
        expected_complete = (stored_base | new2_bool) & ~stored_negative
        if not np.array_equal(complete_bool, expected_complete):
            raise RuntimeError("resumed complete label violates (base OR new2) AND NOT negative")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, artifact in (
            ("edge_vol_sha256", previous_edge_path),
            ("previous_base_sha256", previous_base_path),
            ("new2_sha256", new2_path),
            ("complete_label_sha256", complete_path),
        ):
            if recorded.get(key) != sha256(artifact):
                raise RuntimeError(f"resumed adaptive artifact hash mismatch: {artifact}")
        return AdaptiveMaskResult(
            new2_bool.astype(np.uint8), complete_bool.astype(np.uint8),
            new2_path, complete_path, previous_base_path, manifest_path
        )
    if root.exists():
        raise RuntimeError(f"partial adaptive output exists without manifest: {root}")

    input_dir = root / "input"
    continuous_dir = root / "continuous"
    input_dir.mkdir(parents=True, exist_ok=False)
    edge_path = input_dir / "edge_vol_probability_float32.tif"
    base_path = input_dir / "base_input.tif"
    negative_path = input_dir / "negative_input.tif"
    tifffile.imwrite(edge_path, probability, compression="zlib")
    tifffile.imwrite(base_path, base.astype(np.uint8), compression="zlib")
    tifffile.imwrite(negative_path, negative.astype(np.uint8), compression="zlib")

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(backend)
        if not existing_pythonpath
        else str(backend) + os.pathsep + existing_pythonpath
    )
    continuous_command = [
        sys.executable,
        str(continuous_entry),
        "--edge-vol", str(edge_path),
        "--raw", str(raw),
        "--base", str(base_path),
        "--negative", str(negative_path),
        "--feature-volume", str(feature),
        "--selector", str(selector),
        "--probability-threshold", str(float(probability_threshold)),
        "--top-fraction", str(float(top_fraction)),
        "--trial", str(int(trial)),
        "--run-name", run_name,
        "--output-dir", str(continuous_dir),
    ]
    subprocess.run(continuous_command, check=True, env=environment)
    final_command = [
        sys.executable,
        str(final_entry),
        "--edge-vol", str(edge_path),
        "--raw", str(raw),
        "--base", str(base_path),
        "--negative", str(negative_path),
        "--feature-volume", str(feature),
        "--initial-new2", str(continuous_dir / "test_volume_label_new2.tif"),
        "--frozen-actions", str(actions),
        "--trial", str(int(trial)),
        "--run-name", run_name,
        "--output-dir", str(final_dir),
    ]
    subprocess.run(final_command, check=True, env=environment)

    new2_path = final_dir / "test_volume_label_new2.tif"
    complete_path = final_dir / "test_volume_label_save.tif"
    new2 = np.asarray(tifffile.imread(new2_path) > 0, dtype=bool)
    complete = np.asarray(tifffile.imread(complete_path) > 0, dtype=bool)
    expected = (base | new2) & ~negative
    if np.any(new2 & base) or np.any(new2 & negative):
        raise RuntimeError("adaptive new2 overlaps base or explicit negatives")
    if not np.array_equal(complete, expected):
        raise RuntimeError("adaptive complete label is not (base OR new2) AND NOT negative")
    leaf_summary = _summary_path(final_dir)
    branch_summary = final_dir / "final_adaptive_branch_summary.json"
    manifest = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "trial": int(trial),
        "run_name": run_name,
        "iteration_index_that_generated_new2": int(iteration_index),
        "dense_ground_truth_read": False,
        "edge_line_used": False,
        "complete_label_formula_verified": True,
        "base_voxels": int(base.sum()),
        "new2_voxels": int(new2.sum()),
        "complete_label_voxels": int(complete.sum()),
        "edge_vol_sha256": sha256(edge_path),
        "previous_base_sha256": sha256(base_path),
        "new2_sha256": sha256(new2_path),
        "complete_label_sha256": sha256(complete_path),
        "continuous_materialization_summary": str(
            continuous_dir / "oof_complete_object_new2_summary.json"
        ),
        "final_leaf_materialization_summary": str(leaf_summary),
        "final_branch_summary": str(branch_summary),
        "backend_entry_sha256": {
            continuous_entry.name: sha256(continuous_entry),
            final_entry.name: sha256(final_entry),
        },
        "paper_declared_parameters_changed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return AdaptiveMaskResult(
        new2.astype(np.uint8),
        complete.astype(np.uint8),
        new2_path,
        complete_path,
        base_path,
        manifest_path,
    )


def source_balanced_dataset_class(
    original_dataset: type,
    *,
    backend_dir: str | Path | None,
    new2_path: str | Path,
    previous_base_path: str | Path,
    audit_output_dir: str | Path,
    policy: str,
    seed: int,
) -> type:
    backend = _required_path(
        backend_dir, "SPARSESEG_ADAPTIVE_BACKEND", "adaptive backend directory"
    )
    backend_text = str(backend)
    if backend_text not in sys.path:
        sys.path.insert(0, backend_text)
    sampling = importlib.import_module(
        "run_source_ratio_accumulated_next_iteration_v602"
    )
    sampling.configure_sampling(
        new2_path=Path(new2_path),
        previous_base_path=Path(previous_base_path),
        output_dir=Path(audit_output_dir),
        policy=policy,
        seed=int(seed),
    )
    holder: Any = SimpleNamespace(ValidPatchSliceDataset=original_dataset)
    sampling._install_source_balanced_dataset(holder)
    return holder.ValidPatchSliceDataset
