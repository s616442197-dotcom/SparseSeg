#!/usr/bin/env python3
"""Parallel, GT-free runtime record construction for SparseSeg adaptive new2.

This module is an execution-only optimization of the schema-735 candidate
family. It preserves seed ordering, random-walker parameters, candidate masks,
action ordering, and all 252 selector features. Independent seed walkers and
action statistics are evaluated concurrently, while raw normalization and raw
gradient are shared within each seed crop.
"""

from __future__ import annotations

import os
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Any

import numpy as np
from scipy import ndimage


LAST_PERFORMANCE_AUDIT: dict[str, Any] = {}
LAST_WALKER_AUDIT: dict[str, Any] = {}
_ACTION_PROCESS_CONTEXT: dict[str, Any] = {}
_ACTION_GROUP_ITEMS: list = []
_ACTION_FEATURE_ACCESSOR = None
_CONNECTIVITY_STRUCTURE3 = ndimage.generate_binary_structure(3, 1)


def _seed_connected_fast(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    seed_in_mask = np.asarray(seed, dtype=bool) & mask
    connected = ndimage.binary_propagation(
        seed_in_mask, structure=_CONNECTIVITY_STRUCTURE3, mask=mask
    )
    if os.environ.get("SPARSESEG_VERIFY_CONNECTIVITY_FAST", "0") == "1":
        labels, _ = ndimage.label(mask, structure=_CONNECTIVITY_STRUCTURE3)
        connected_ids = np.unique(labels[np.asarray(seed, dtype=bool)])
        connected_ids = connected_ids[connected_ids > 0]
        reference = np.isin(labels, connected_ids)
        if not np.array_equal(connected, reference):
            raise RuntimeError(
                "binary_propagation connectivity differs from label/isin"
            )
    return np.asarray(connected, dtype=bool)


def _xy_dilate_fast(mask: np.ndarray, radius: int) -> np.ndarray:
    """Exact rectangular XY dilation via two separable 1-D max filters."""
    value = np.asarray(mask, dtype=np.uint8)
    size = 2 * int(radius) + 1
    value = ndimage.maximum_filter1d(
        value, size=size, axis=1, mode="constant", cval=0
    )
    value = ndimage.maximum_filter1d(
        value, size=size, axis=2, mode="constant", cval=0
    )
    result = np.asarray(value > 0, dtype=bool)
    if os.environ.get("SPARSESEG_VERIFY_SEPARABLE_MORPH", "0") == "1":
        structure = np.ones((1, size, size), dtype=bool)
        reference = ndimage.binary_dilation(mask, structure=structure)
        if not np.array_equal(result, reference):
            raise RuntimeError(
                "separable XY dilation differs from rectangular dilation"
            )
    return result


def _xy_close_fast(mask: np.ndarray, radius: int) -> np.ndarray:
    """Exact rectangular XY closing via separable max/min filters."""
    size = 2 * int(radius) + 1
    value = _xy_dilate_fast(mask, radius).astype(np.uint8, copy=False)
    value = ndimage.minimum_filter1d(
        value, size=size, axis=1, mode="constant", cval=0
    )
    value = ndimage.minimum_filter1d(
        value, size=size, axis=2, mode="constant", cval=0
    )
    result = np.asarray(value > 0, dtype=bool)
    if os.environ.get("SPARSESEG_VERIFY_SEPARABLE_MORPH", "0") == "1":
        structure = np.ones((1, size, size), dtype=bool)
        reference = ndimage.binary_closing(mask, structure=structure)
        if not np.array_equal(result, reference):
            raise RuntimeError(
                "separable XY closing differs from rectangular closing"
            )
    return result


def _close_fill_xy_fast(profile: Any, mask: np.ndarray, radius: int) -> np.ndarray:
    value = _xy_close_fast(mask, radius) | mask
    return profile.alignment.fill_xy(value)


def _worker_count() -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    requested = int(os.environ.get("SPARSESEG_MASK_WORKERS", allocated))
    return max(1, min(requested, allocated))


def _candidate_runtime_features(
    generator: Any,
    candidate: np.ndarray,
    seed: np.ndarray,
    p_local: np.ndarray,
    raw_local: np.ndarray,
    raw_gradient: np.ndarray,
) -> dict[str, float]:
    ring = _xy_dilate_fast(candidate, 2) & ~candidate
    raw_inside = float(np.mean(raw_local[candidate]))
    raw_ring = float(np.mean(raw_local[ring])) if np.any(ring) else raw_inside
    boundary = _xy_dilate_fast(candidate, 1) & ~ndimage.binary_erosion(
        candidate, structure=np.ones((1, 3, 3), dtype=bool)
    )
    return {
        "seed_probability_mean": float(np.mean(p_local[seed])),
        "seed_probability_min": float(np.min(p_local[seed])),
        "region_probability_mean": float(np.mean(p_local[candidate])),
        "region_probability_q90": float(np.quantile(p_local[candidate], 0.90)),
        "raw_inside_mean": raw_inside,
        "raw_ring_mean": raw_ring,
        "raw_inside_ring_contrast": abs(raw_inside - raw_ring),
        "boundary_gradient_mean": float(np.mean(raw_gradient[boundary])),
    }


def _parallel_source_records(
    probability: np.ndarray,
    raw: np.ndarray,
    base: np.ndarray,
    negative: np.ndarray,
    profile: Any,
    workers: int,
) -> tuple[list[dict], dict, dict[float, np.ndarray], float]:
    generator = profile.generator
    structure3 = ndimage.generate_binary_structure(3, 1)
    seed_summary: dict[str, int] = {}
    label_maps: dict[float, np.ndarray] = {}
    tasks: list[tuple[float, int, tuple[slice, ...], np.ndarray]] = []

    for seed_threshold in generator.SEED_THRESHOLDS:
        seeds = (probability >= seed_threshold) & ~base & ~negative
        labels, count = ndimage.label(seeds, structure=structure3)
        label_maps[float(seed_threshold)] = labels
        objects = ndimage.find_objects(labels, max_label=count)
        seed_summary[f"{seed_threshold:.2f}"] = int(count)
        for seed_id in range(1, count + 1):
            bounds = objects[seed_id - 1]
            if bounds is None:
                continue
            crop = generator.crop_bounds(bounds, probability.shape)
            tasks.append((float(seed_threshold), int(seed_id), crop, labels))

    started = time.perf_counter()

    def run_seed(task):
        seed_threshold, seed_id, crop, labels = task
        seed = labels[crop] == seed_id
        seed_voxels = int(seed.sum())
        if seed_voxels < min(generator.MINIMUM_SEED_VOXELS):
            return []
        p_local = probability[crop]
        raw_local = generator.normalize_raw(raw[crop])
        raw_gradient = ndimage.gaussian_gradient_magnitude(
            raw_local, sigma=(0.5, 1.0, 1.0)
        )
        negative_local = negative[crop]
        markers = np.zeros(seed.shape, dtype=np.uint8)
        markers[seed] = 2
        background = generator.boundary_mask(seed.shape) | negative_local
        background &= ~seed
        markers[background] = 1
        if not np.any(markers == 1):
            return []

        result = []
        for beta in generator.BETAS:
            try:
                full_probability = generator.random_walker(
                    raw_local,
                    markers,
                    beta=beta,
                    mode="cg_j",
                    spacing=(4.0, 1.0, 1.0),
                    return_full_prob=True,
                )
            except Exception as error:
                print(
                    f"[parallel-walker-failed] seed-thr={seed_threshold} "
                    f"seed={seed_id} beta={beta}: {error}",
                    flush=True,
                )
                continue
            foreground_probability = np.asarray(full_probability[1], dtype=np.float32)
            for walker_threshold in generator.WALKER_THRESHOLDS:
                candidate = foreground_probability >= walker_threshold
                candidate |= seed
                candidate = _xy_close_fast(candidate, 1) | seed
                candidate &= ~base[crop] & ~negative_local
                candidate = _seed_connected_fast(candidate, seed)
                candidate &= ~base[crop] & ~negative_local
                total = int(candidate.sum())
                if total == 0:
                    continue
                candidate_objects = ndimage.find_objects(
                    candidate.astype(np.uint8), max_label=1
                )
                candidate_bounds = candidate_objects[0] if candidate_objects else None
                if candidate_bounds is None:
                    continue
                z_span = int(
                    candidate_bounds[0].stop - candidate_bounds[0].start
                )
                touches_crop_boundary = bool(
                    np.any(candidate[0])
                    or np.any(candidate[-1])
                    or np.any(candidate[:, 0])
                    or np.any(candidate[:, -1])
                    or np.any(candidate[:, :, 0])
                    or np.any(candidate[:, :, -1])
                )
                public = {
                    "seed_threshold": seed_threshold,
                    "seed_id": seed_id,
                    "seed_voxels": seed_voxels,
                    "beta": beta,
                    "walker_threshold": walker_threshold,
                    "voxels": total,
                    "growth_ratio": total / seed_voxels,
                    "z_span": z_span,
                    "touches_crop_boundary": touches_crop_boundary,
                }
                public.update(
                    _candidate_runtime_features(
                        generator,
                        candidate,
                        seed,
                        p_local,
                        raw_local,
                        raw_gradient,
                    )
                )
                public.update(
                    {
                        "true_positive_voxels": 0,
                        "false_positive_voxels": total,
                        "far_false_positive_voxels": total,
                        "precision": 0.0,
                    }
                )
                result.append(
                    {
                        "public": public,
                        "crop": crop,
                        "_packed_mask": np.packbits(candidate, axis=None),
                        "_mask_shape": tuple(candidate.shape),
                    }
                )
        return result

    global LAST_WALKER_AUDIT
    records: list[dict] = []
    process_backend = (
        os.environ.get("SPARSESEG_WALKER_BACKEND", "process").lower()
        == "process"
        and "fork" in multiprocessing.get_all_start_methods()
    )
    if process_backend:
        fork_context = multiprocessing.get_context("fork")
        task_queue = fork_context.Queue()
        result_queue = fork_context.Queue()
        completed_families = [None] * len(tasks)

        def process_loop():
            while True:
                task_index = task_queue.get()
                if task_index is None:
                    return
                try:
                    family = run_seed(tasks[task_index])
                    result_queue.put((task_index, family, None))
                except BaseException as error:
                    result_queue.put((task_index, None, repr(error)))

        processes = [
            fork_context.Process(target=process_loop)
            for _ in range(workers)
        ]
        for process in processes:
            process.start()
        for task_index in range(len(tasks)):
            task_queue.put(task_index)
        for _ in processes:
            task_queue.put(None)

        received_sources = 0
        errors = []
        for completed_count in range(1, len(tasks) + 1):
            task_index, family, error = result_queue.get()
            if error is not None:
                errors.append((task_index, error))
                completed_families[task_index] = []
            else:
                completed_families[task_index] = family
                received_sources += len(family)
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(
                    f"[process-walker] completed={completed_count}/{len(tasks)} "
                    f"sources={received_sources}",
                    flush=True,
                )
        for process in processes:
            process.join()
        exit_failures = [
            process.exitcode for process in processes
            if process.exitcode not in (0, None)
        ]
        task_queue.close()
        result_queue.close()
        if errors or exit_failures:
            raise RuntimeError(
                f"walker process failures: tasks={errors[:3]}, "
                f"exit_codes={exit_failures}"
            )
        for family in completed_families:
            records.extend(family)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, family in enumerate(pool.map(run_seed, tasks), 1):
                records.extend(family)
                if index % 25 == 0 or index == len(tasks):
                    print(
                        f"[parallel-walker] seed={index}/{len(tasks)} "
                        f"sources={len(records)}",
                        flush=True,
                    )
    LAST_WALKER_AUDIT = {
        "backend": "process" if process_backend else "thread",
        "worker_count": workers,
        "seed_task_count": len(tasks),
        "record_order_preserved": True,
    }
    return records, seed_summary, label_maps, time.perf_counter() - started


def _feature_standardization_context(profile: Any, feature: np.ndarray) -> dict:
    spatial_sample = profile.feature_profile.bounded_sample(
        feature.reshape(-1, feature.shape[-1]), 100000
    )
    q25, median, q75 = np.quantile(
        spatial_sample, (0.25, 0.50, 0.75), axis=0
    )
    standard_deviation = np.std(spatial_sample, axis=0)
    scale = np.maximum(q75 - q25, 0.10 * standard_deviation)
    valid = np.isfinite(scale) & (scale > 1e-7)
    return {
        "median": median,
        "scale": scale,
        "valid_indices": np.flatnonzero(valid),
    }


def _standardized_feature_distance(
    profile: Any,
    feature: np.ndarray,
    core: np.ndarray,
    standardization: dict,
) -> tuple[np.ndarray, float, int]:
    valid_indices = standardization["valid_indices"]
    if valid_indices.size == 0:
        return np.zeros(feature.shape[:-1], dtype=np.float32), 0.0, 0
    core_values = profile.feature_profile.bounded_sample(feature[core], 50000)
    prototype = (
        np.median(core_values, axis=0)
        if core_values.size
        else standardization["median"]
    )
    scale = np.asarray(
        standardization["scale"][valid_indices], dtype=np.float32
    )
    feature_values = np.asarray(
        feature[..., valid_indices], dtype=np.float32
    )
    prototype_values = np.asarray(
        prototype[valid_indices], dtype=np.float32
    )
    delta = (feature_values - prototype_values) / scale
    squared = np.square(
        np.clip(delta, -10.0, 10.0), dtype=np.float32
    )
    distance = np.add.reduce(
        squared, axis=-1, dtype=np.float32
    )
    distance /= float(valid_indices.size)
    core_distance = distance[core]
    reference = (
        float(np.quantile(core_distance, 0.95))
        if core_distance.size
        else 0.0
    )
    return distance, max(reference, 0.50), int(valid_indices.size)


def _action_variants(
    profile: Any,
    candidate: np.ndarray,
    seed: np.ndarray,
    probability: np.ndarray,
    feature: np.ndarray,
    allowed: np.ndarray,
    standardization: dict,
    distance_cache: dict | None = None,
):
    candidate = np.asarray(candidate, dtype=bool) & allowed
    seed = np.asarray(seed, dtype=bool) & allowed
    actions = [("raw_random_walker", {}, candidate)]

    for radius in (1, 2, 3):
        completed = _close_fill_xy_fast(
            profile, candidate, radius
        ) & allowed
        completed = _seed_connected_fast(
            (completed | seed) & allowed, seed
        ) & allowed
        actions.append(
            ("morphology_close_fill", {"close_radius": radius}, completed)
        )

    core = seed | (candidate & (probability >= 0.50))
    core_key = (
        int(core.sum()),
        np.packbits(core, axis=None).tobytes(),
    )
    cached_distance = (
        distance_cache.get(core_key) if distance_cache is not None else None
    )
    if cached_distance is None:
        cached_distance = _standardized_feature_distance(
            profile, feature, core, standardization
        )
        if distance_cache is not None:
            distance_cache[core_key] = cached_distance
    distance, reference, valid_channels = cached_distance
    for radius, probability_threshold, multiplier, close_radius in profile.FEATURE_ACTIONS:
        spatial_guard = _xy_dilate_fast(candidate, radius) & allowed
        support = (
            spatial_guard
            & (probability >= probability_threshold)
            & (distance <= reference * multiplier)
        ) | candidate | seed
        grown = _seed_connected_fast(
            support & allowed, seed
        ) & allowed
        grown = _close_fill_xy_fast(
            profile, grown, close_radius
        ) & allowed
        grown = _seed_connected_fast(
            (grown | seed) & allowed, seed
        ) & allowed
        actions.append(
            (
                "feature_probability_object_completion",
                {
                    "feature_radius": radius,
                    "feature_probability_threshold": probability_threshold,
                    "feature_distance_multiplier": multiplier,
                    "feature_close_radius": close_radius,
                    "feature_valid_channel_count": valid_channels,
                    "feature_core_distance_q95": reference,
                },
                grown,
            )
        )

    unique = []
    seen = set()
    for name, parameters, value in actions:
        key = (int(value.sum()), np.packbits(value, axis=None).tobytes())
        if key not in seen and np.any(value):
            unique.append((name, parameters, value))
            seen.add(key)
    return unique


def _score_action(
    profile: Any,
    source: dict,
    action_name: str,
    parameters: dict,
    candidate: np.ndarray,
    context: dict,
) -> dict:
    generator = profile.generator
    crop = source["crop"]
    p_local = context["probability"]
    raw_local = context["raw"]
    raw_gradient = context["raw_gradient"]
    seed = context["seed"]
    base_local = context["base"]
    negative_local = context["negative"]
    feature = context["feature"]
    allowed = context["allowed"]

    total = int(candidate.sum())
    bounds = profile.mask_bounds(candidate)
    if bounds is None:
        raise RuntimeError("Nonempty complete-object action has no bounds")
    z_span = int(bounds[0].stop - bounds[0].start)
    runtime = _candidate_runtime_features(
        generator, candidate, seed, p_local, raw_local, raw_gradient
    )
    public = {
        **source["public"],
        "action": action_name,
        "action_code": {
            "raw_random_walker": 0,
            "morphology_close_fill": 1,
            "feature_probability_object_completion": 2,
        }[action_name],
        **parameters,
        "voxels": total,
        "growth_ratio": total / max(1, int(source["public"]["seed_voxels"])),
        "z_span": z_span,
        "touches_crop_boundary": profile.touches_boundary(candidate),
        "true_positive_voxels": 0,
        "false_positive_voxels": total,
        "far_false_positive_voxels": total,
        "precision": 0.0,
    }
    public.update(runtime)

    inner = _xy_dilate_fast(candidate, 1)
    outer = _xy_dilate_fast(candidate, 3)
    ring = outer & ~inner & ~base_local & ~negative_local
    ring_probability = p_local[ring]
    if ring_probability.size:
        ring_mean = float(np.mean(ring_probability))
        ring_q50 = float(np.quantile(ring_probability, 0.50))
        ring_q90 = float(np.quantile(ring_probability, 0.90))
        ring_max = float(np.max(ring_probability))
        ring_frac_p10 = float(np.mean(ring_probability >= 0.10))
        ring_frac_p20 = float(np.mean(ring_probability >= 0.20))
        ring_frac_p50 = float(np.mean(ring_probability >= 0.50))
        ring_raw_mean = float(np.mean(raw_local[ring]))
    else:
        ring_mean = ring_q50 = ring_q90 = ring_max = 0.0
        ring_frac_p10 = ring_frac_p20 = ring_frac_p50 = 0.0
        ring_raw_mean = public["raw_inside_mean"]
    public.update(
        {
            "loss_ring_voxels": int(ring.sum()),
            "loss_ring_probability_mean": ring_mean,
            "loss_ring_probability_q50": ring_q50,
            "loss_ring_probability_q90": ring_q90,
            "loss_ring_probability_max": ring_max,
            "loss_ring_probability_fraction_ge_0_10": ring_frac_p10,
            "loss_ring_probability_fraction_ge_0_20": ring_frac_p20,
            "loss_ring_probability_fraction_ge_0_50": ring_frac_p50,
            "loss_ring_raw_mean": ring_raw_mean,
            "loss_ring_raw_inside_contrast": abs(
                public["raw_inside_mean"] - ring_raw_mean
            ),
            "local_induced_area_negative_gt_voxels": 0,
            "local_relieved_area_negative_gt_voxels": 0,
            "local_development_loss_aware_gain": -10 * total,
        }
    )
    public.update(
        profile.feature_profile.candidate_feature_statistics(
            feature, candidate, allowed
        )
    )
    public.update(
        {
            "matched_gt_component": 0,
            "matched_gt_intersection_voxels": 0,
            "matched_gt_available_voxels": 0,
            "matched_gt_precision": 0.0,
            "matched_gt_recall": 0.0,
            "overlapping_gt_component_count": 0,
        }
    )
    public["induced_ring_fraction_of_tp"] = 0.0
    public["object_complete_development_target"] = False
    return public



def _process_seed_group(group_index: int):
    """Generate and score one complete seed family in a forked worker.

    Large arrays and packed source records are inherited copy-on-write. Only
    an integer task index and final packed records cross process queues.
    """
    global _ACTION_FEATURE_ACCESSOR
    context_root = _ACTION_PROCESS_CONTEXT
    profile = context_root["profile"]
    key, family = _ACTION_GROUP_ITEMS[group_index]
    seed_threshold, seed_id = key
    crop = family[0]["crop"]
    if any(source["crop"] != crop for source in family):
        raise RuntimeError("same seed unexpectedly has different crops")

    if _ACTION_FEATURE_ACCESSOR is None:
        _ACTION_FEATURE_ACCESSOR = profile.feature_profile.FeatureVolumeAccessor(
            context_root["feature_volume"], cache_size=1
        )
    feature = _ACTION_FEATURE_ACCESSOR.get(crop)
    labels = context_root["label_maps"][seed_threshold]
    seed = labels[crop] == seed_id
    raw_local = profile.generator.normalize_raw(context_root["raw"][crop])
    seed_context = {
        "seed": seed,
        "feature": feature,
        "allowed": ~(
            context_root["base"][crop] | context_root["negative"][crop]
        ),
        "probability": context_root["probability"][crop],
        "raw": raw_local,
        "raw_gradient": ndimage.gaussian_gradient_magnitude(
            raw_local, sigma=(0.5, 1.0, 1.0)
        ),
        "base": context_root["base"][crop],
        "negative": context_root["negative"][crop],
    }
    standardization = _feature_standardization_context(profile, feature)

    generated_started = time.perf_counter()
    source_actions = []
    distance_cache = {}
    for source in family:
        source["seed"] = seed
        source_mask = np.unpackbits(
            source["_packed_mask"], count=int(np.prod(source["_mask_shape"]))
        ).reshape(source["_mask_shape"]).astype(bool, copy=False)
        source_actions.append(
            (
                source,
                _action_variants(
                    profile,
                    source_mask,
                    seed,
                    seed_context["probability"],
                    feature,
                    seed_context["allowed"],
                    standardization,
                    distance_cache,
                ),
            )
        )
    generation_seconds = time.perf_counter() - generated_started

    scored_started = time.perf_counter()
    records = []
    for source, actions in source_actions:
        for action_name, parameters, candidate in actions:
            public = _score_action(
                profile,
                source,
                action_name,
                parameters,
                candidate,
                seed_context,
            )
            records.append(
                {
                    "public": public,
                    "crop": source["crop"],
                    "packed_mask": np.packbits(candidate, axis=None),
                    "mask_shape": tuple(candidate.shape),
                }
            )
    scoring_seconds = time.perf_counter() - scored_started
    return (
        records,
        generation_seconds,
        scoring_seconds,
        len(family),
        len(distance_cache),
    )


def construct_runtime_records(
    probability: np.ndarray,
    raw: np.ndarray,
    base: np.ndarray,
    negative: np.ndarray,
    feature_volume,
    profile: Any,
) -> tuple[list[dict], dict]:
    global LAST_PERFORMANCE_AUDIT
    started = time.perf_counter()
    workers = _worker_count()
    source_records, seed_summary, label_maps, walker_seconds = (
        _parallel_source_records(
            probability, raw, base, negative, profile, workers
        )
    )
    feature_accessor = profile.feature_profile.FeatureVolumeAccessor(
        feature_volume, cache_size=1
    )
    if feature_accessor.spatial_shape != probability.shape:
        raise ValueError("Feature and edge-vol shapes differ")

    groups: OrderedDict[tuple[float, int], list[dict]] = OrderedDict()
    for source in source_records:
        key = (
            float(source["public"]["seed_threshold"]),
            int(source["public"]["seed_id"]),
        )
        groups.setdefault(key, []).append(source)

    records: list[dict] = []
    action_generation_seconds = 0.0
    action_scoring_seconds = 0.0
    batch_size = max(1, int(os.environ.get("SPARSESEG_MASK_SEED_BATCH", "8")))
    group_items = list(groups.items())

    process_backend = (
        os.environ.get("SPARSESEG_MASK_BACKEND", "process").lower()
        == "process"
        and "fork" in multiprocessing.get_all_start_methods()
    )
    if process_backend:
        global _ACTION_PROCESS_CONTEXT, _ACTION_GROUP_ITEMS
        _ACTION_PROCESS_CONTEXT = {
            "probability": probability,
            "raw": raw,
            "base": base,
            "negative": negative,
            "feature_volume": feature_volume,
            "profile": profile,
            "label_maps": label_maps,
        }
        _ACTION_GROUP_ITEMS = group_items
        process_started = time.perf_counter()
        distance_cache_requests = 0
        distance_cache_misses = 0
        try:
            fork_context = multiprocessing.get_context("fork")
            with fork_context.Pool(processes=workers) as pool:
                for index, result in enumerate(
                    pool.imap(_process_seed_group, range(len(group_items))), 1
                ):
                    (
                        family_records,
                        generation_seconds,
                        scoring_seconds,
                        family_cache_requests,
                        family_cache_misses,
                    ) = result
                    records.extend(family_records)
                    distance_cache_requests += family_cache_requests
                    distance_cache_misses += family_cache_misses
                    action_generation_seconds += generation_seconds
                    action_scoring_seconds += scoring_seconds
                    if index % 25 == 0 or index == len(group_items):
                        print(
                            f"[process-oof] seed={index}/{len(group_items)} "
                            f"actions={len(records)}",
                            flush=True,
                        )
        finally:
            _ACTION_PROCESS_CONTEXT = {}
            _ACTION_GROUP_ITEMS = []
        process_seconds = time.perf_counter() - process_started
        if not records:
            raise RuntimeError("GT-free runtime candidate family is empty")
        LAST_PERFORMANCE_AUDIT = {
            "implementation": "forked_complete_seed_context_v800",
            "backend": "process",
            "worker_count": workers,
            "seed_batch_size": None,
            "source_seed_count": len(group_items),
            "source_record_count": len(source_records),
            "candidate_action_count": len(records),
            "walker_wall_clock_seconds": walker_seconds,
            "walker_runtime_audit": dict(LAST_WALKER_AUDIT),
            "action_process_wall_clock_seconds": process_seconds,
            "summed_worker_action_generation_seconds": action_generation_seconds,
            "summed_worker_action_scoring_seconds": action_scoring_seconds,
            "feature_distance_cache_requests": distance_cache_requests,
            "feature_distance_cache_misses": distance_cache_misses,
            "feature_distance_cache_hits": (
                distance_cache_requests - distance_cache_misses
            ),
            "connectivity_implementation": "binary_propagation_from_seed",
            "connectivity_reference_verified": (
                os.environ.get("SPARSESEG_VERIFY_CONNECTIVITY_FAST", "0") == "1"
            ),
            "xy_rectangular_morphology_implementation": (
                "separable_maximum_minimum_filter1d"
            ),
            "xy_rectangular_morphology_reference_verified": (
                os.environ.get("SPARSESEG_VERIFY_SEPARABLE_MORPH", "0") == "1"
            ),
            "total_construct_runtime_records_wall_clock_seconds": (
                time.perf_counter() - started
            ),
            "candidate_and_feature_definitions_changed": False,
            "record_order_preserved": True,
        }
        return records, seed_summary

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(group_items), batch_size):
            batch = group_items[offset : offset + batch_size]
            contexts: dict[tuple[float, int], dict] = {}
            action_inputs = []
            for key, family in batch:
                seed_threshold, seed_id = key
                crop = family[0]["crop"]
                if any(source["crop"] != crop for source in family):
                    raise RuntimeError("same seed unexpectedly has different crops")
                labels = label_maps[seed_threshold]
                seed = labels[crop] == seed_id
                feature = feature_accessor.get(crop)
                context = {
                    "seed": seed,
                    "feature": feature,
                    "allowed": ~(base[crop] | negative[crop]),
                    "probability": probability[crop],
                    "raw_input": raw[crop],
                    "base": base[crop],
                    "negative": negative[crop],
                }
                contexts[key] = context
                for source in family:
                    source["seed"] = seed
                    source_mask = np.unpackbits(
                        source["_packed_mask"],
                        count=int(np.prod(source["_mask_shape"])),
                    ).reshape(source["_mask_shape"]).astype(bool, copy=False)
                    action_inputs.append((key, source, source_mask))

            def prepare(key):
                context = contexts[key]
                raw_local = profile.generator.normalize_raw(
                    context["raw_input"]
                )
                return (
                    key,
                    raw_local,
                    ndimage.gaussian_gradient_magnitude(
                        raw_local, sigma=(0.5, 1.0, 1.0)
                    ),
                    _feature_standardization_context(
                        profile, context["feature"]
                    ),
                )

            for key, raw_local, raw_gradient, standardization in pool.map(
                prepare, list(contexts)
            ):
                contexts[key]["raw"] = raw_local
                contexts[key]["raw_gradient"] = raw_gradient
                contexts[key]["feature_standardization"] = standardization
                contexts[key].pop("raw_input", None)

            generated_started = time.perf_counter()

            def generate(item):
                key, source, source_mask = item
                context = contexts[key]
                return _action_variants(
                    profile,
                    source_mask,
                    context["seed"],
                    context["probability"],
                    context["feature"],
                    context["allowed"],
                    context["feature_standardization"],
                )

            generated = list(pool.map(generate, action_inputs))
            action_generation_seconds += time.perf_counter() - generated_started

            score_inputs = []
            for (key, source, _), actions in zip(action_inputs, generated):
                for action_name, parameters, candidate in actions:
                    score_inputs.append(
                        (key, source, action_name, parameters, candidate)
                    )

            scored_started = time.perf_counter()

            def score(item):
                key, source, action_name, parameters, candidate = item
                public = _score_action(
                    profile,
                    source,
                    action_name,
                    parameters,
                    candidate,
                    contexts[key],
                )
                return {
                    "public": public,
                    "crop": source["crop"],
                    "packed_mask": np.packbits(candidate, axis=None),
                    "mask_shape": tuple(candidate.shape),
                }

            records.extend(pool.map(score, score_inputs))
            action_scoring_seconds += time.perf_counter() - scored_started
            completed = min(offset + len(batch), len(group_items))
            print(
                f"[parallel-oof] seed={completed}/{len(group_items)} "
                f"actions={len(records)}",
                flush=True,
            )

    if not records:
        raise RuntimeError("GT-free runtime candidate family is empty")
    LAST_PERFORMANCE_AUDIT = {
        "implementation": "parallel_shared_seed_context_v800",
        "worker_count": workers,
        "seed_batch_size": batch_size,
        "source_seed_count": len(group_items),
        "source_record_count": len(source_records),
        "candidate_action_count": len(records),
        "walker_wall_clock_seconds": walker_seconds,
        "walker_runtime_audit": dict(LAST_WALKER_AUDIT),
        "action_generation_wall_clock_seconds": action_generation_seconds,
        "action_scoring_wall_clock_seconds": action_scoring_seconds,
        "total_construct_runtime_records_wall_clock_seconds": (
            time.perf_counter() - started
        ),
        "candidate_and_feature_definitions_changed": False,
        "record_order_preserved": True,
    }
    return records, seed_summary
