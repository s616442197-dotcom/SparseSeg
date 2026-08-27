#!/usr/bin/env python3
"""Cached continuous complete-object policy with unseen-trial LOTO ensembling.

Known development trials use their exact held-out model. New trial identifiers
use the mean of all stored LOTO models, without reading their dense GT.
"""

from pathlib import Path


SOURCE = Path(__file__).with_name("materialize_oof_complete_object_new2_v612.py")
source = SOURCE.read_text(encoding="utf-8")
replacements = {
    '    parser.add_argument("--trial", type=int, choices=(100, 101, 102), required=True)':
        '    parser.add_argument("--trial", type=int, required=True)',
    "    parser.add_argument(\"--ground-truth\", type=Path)\n"
    "    return parser.parse_args()": (
        "    parser.add_argument(\"--ground-truth\", type=Path)\n"
        "    parser.add_argument(\"--probability-threshold\", type=float, required=True)\n"
        "    parser.add_argument(\"--top-fraction\", type=float, required=True)\n"
        "    return parser.parse_args()"
    ),
    "    zero_gt = np.zeros(base.shape, dtype=bool)\n"
    "    feature_accessor = profile.feature_profile.FeatureVolumeAccessor(feature_volume)": (
        "    zero_gt = np.zeros(base.shape, dtype=bool)\n"
        "    zero_labels = np.zeros(base.shape, dtype=np.uint8)\n"
        "    feature_accessor = profile.feature_profile.FeatureVolumeAccessor(feature_volume)"
    ),
    "                np.zeros(base.shape, dtype=np.uint8),": "                zero_labels,",
    "        seed = seed_labels[crop] == int(source[\"public\"][\"seed_id\"])\n"
    "        feature = feature_accessor.get(crop)": (
        "        seed = seed_labels[crop] == int(source[\"public\"][\"seed_id\"])\n"
        "        source[\"seed\"] = seed\n"
        "        feature = feature_accessor.get(crop)"
    ),
    "def select_runtime_records(\n"
    "    records: list[dict], selector_path: Path, trial: int\n"
    ") -> tuple[list[dict], dict]:": (
        "def select_runtime_records(\n"
        "    records: list[dict], selector_path: Path, trial: int,\n"
        "    probability_threshold: float, top_fraction: float,\n"
        ") -> tuple[list[dict], dict]:"
    ),
    "    policy = dict(payload[\"best_policy\"])": (
        "    if not 0.0 <= probability_threshold <= 1.0:\n"
        "        raise ValueError(\"probability threshold must be in [0, 1]\")\n"
        "    if not 0.0 < top_fraction <= 1.0:\n"
        "        raise ValueError(\"top fraction must be in (0, 1]\")\n"
        "    policy = {\n"
        "        \"probability_threshold\": float(probability_threshold),\n"
        "        \"top_fraction\": float(top_fraction),\n"
        "        \"source\": \"continuous_LOTO_OOF_search_schema_623\",\n"
        "    }"
    ),
    '''    models = payload["heldout_models"]
    model_pair = models.get(trial, models.get(str(trial)))
    if model_pair is None:
        raise RuntimeError(f"Selector has no held-out model for trial {trial}")

    frame = pd.DataFrame([record["public"] for record in records])
    matrix = numeric_matrix(frame, features)
    classifier = model_pair["classifier"]
    regressor = model_pair["regressor"]
    positive = np.flatnonzero(np.asarray(classifier.classes_) == 1)
    if positive.size != 1:
        raise RuntimeError("Held-out classifier lacks exactly one positive class")
    probability = classifier.predict_proba(matrix)[:, int(positive[0])]
    predicted_log_tp = regressor.predict(matrix)
    predicted_tp = np.expm1(np.maximum(predicted_log_tp, 0.0))
''': '''    models = payload["heldout_models"]
    model_pair = models.get(trial, models.get(str(trial)))
    if model_pair is None:
        model_pairs = [models[key] for key in sorted(models, key=lambda value: int(value))]
        model_identity = "mean_of_all_stored_LOTO_models_for_unseen_trial"
    else:
        model_pairs = [model_pair]
        model_identity = trial

    frame = pd.DataFrame([record["public"] for record in records])
    matrix = numeric_matrix(frame, features)
    probabilities = []
    predicted_tps = []
    for current_pair in model_pairs:
        classifier = current_pair["classifier"]
        regressor = current_pair["regressor"]
        positive = np.flatnonzero(np.asarray(classifier.classes_) == 1)
        if positive.size != 1:
            raise RuntimeError("Held-out classifier lacks exactly one positive class")
        probabilities.append(
            classifier.predict_proba(matrix)[:, int(positive[0])]
        )
        predicted_log_tp = regressor.predict(matrix)
        predicted_tps.append(np.expm1(np.maximum(predicted_log_tp, 0.0)))
    probability = np.mean(np.stack(probabilities), axis=0)
    predicted_tp = np.mean(np.stack(predicted_tps), axis=0)
''',
    '        "heldout_trial_model": trial,': '        "heldout_trial_model": model_identity,',
    "    selected, selection_audit = select_runtime_records(\n"
    "        records, args.selector, args.trial\n"
    "    )": (
        "    selected, selection_audit = select_runtime_records(\n"
        "        records, args.selector, args.trial,\n"
        "        args.probability_threshold, args.top_fraction,\n"
        "    )"
    ),
    '"schema_version": 612,': '"schema_version": 718,',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"v612 source changed; replacement not unique: {old}")
    source = source.replace(old, new)

# Preserve the selector's established zero encoding when every runtime action
# omits the optional close_radius field; keep strict failures for other fields.
missing_close_radius_old = '''        if name not in frame:
            raise RuntimeError(f"Runtime candidate table lacks selector feature {name}")
        value = frame[name]
'''
missing_close_radius_new = '''        if name not in frame:
            if name == "close_radius":
                columns.append(np.zeros(len(frame), dtype=np.float32))
                continue
            raise RuntimeError(f"Runtime candidate table lacks selector feature {name}")
        value = frame[name]
'''
if source.count(missing_close_radius_old) != 1:
    raise RuntimeError("v735 optional close_radius encoding anchor changed")
source = source.replace(missing_close_radius_old, missing_close_radius_new)


# Performance-equivalent deployment optimization: all walker thresholds for a
# seed share one crop. Reuse the seed mask, feature crop and allowed mask rather
# than rereading the same 36-channel feature block five times.
cache_old = '''    available_sizes = np.zeros(1, dtype=np.int64)
    records = []
    for index, source in enumerate(source_records, 1):
        crop = source["crop"]
        seed = seed_labels[crop] == int(source["public"]["seed_id"])
        source["seed"] = seed
        feature = feature_accessor.get(crop)
        allowed_local = ~(base[crop] | negative[crop])
'''
cache_new = '''    available_sizes = np.zeros(1, dtype=np.int64)
    records = []
    cached_seed_id = None
    cached_seed = None
    cached_feature = None
    cached_allowed_local = None
    cached_crop = None
    feature_crop_cache_hits = 0
    feature_crop_cache_misses = 0
    for index, source in enumerate(source_records, 1):
        crop = source["crop"]
        seed_id = int(source["public"]["seed_id"])
        if cached_seed_id != seed_id:
            cached_seed_id = seed_id
            cached_crop = crop
            cached_seed = seed_labels[crop] == seed_id
            cached_feature = feature_accessor.get(crop)
            cached_allowed_local = ~(base[crop] | negative[crop])
            feature_crop_cache_misses += 1
        else:
            if crop != cached_crop:
                raise RuntimeError("same seed unexpectedly has different crops")
            feature_crop_cache_hits += 1
        seed = cached_seed
        feature = cached_feature
        allowed_local = cached_allowed_local
        source["seed"] = seed
'''
if source.count(cache_old) != 1:
    raise RuntimeError("v718 transformed construct_runtime_records block changed")
source = source.replace(cache_old, cache_new)

# Score the action variants of one source concurrently.  The action list,
# executor.map return order, selector matrix rows and union order remain exact.
if source.count("import argparse\n") != 1:
    raise RuntimeError("v718 generated import block changed")
source = source.replace(
    "import argparse\n",
    "import argparse\nfrom concurrent.futures import ThreadPoolExecutor\n",
)
pool_old = '''    feature_crop_cache_misses = 0
    for index, source in enumerate(source_records, 1):
'''
pool_new = '''    feature_crop_cache_misses = 0
    score_pool = ThreadPoolExecutor(max_workers=4)
    for index, source in enumerate(source_records, 1):
'''
if source.count(pool_old) != 1:
    raise RuntimeError("v735 score-pool insertion anchor changed")
source = source.replace(pool_old, pool_new)
score_old = '''        for action_name, parameters, candidate in profile.action_variants(
            source["mask"],
            seed,
            probability[crop],
            feature,
            allowed_local,
        ):
            public = profile.score_action(
                source,
                action_name,
                parameters,
                candidate,
                probability,
                raw,
                base,
                negative,
                zero_gt,
                zero_gt,
                zero_labels,
                available_sizes,
                feature,
            )
            records.append({"public": public, "crop": crop, "mask": candidate})
'''
score_new = '''        actions = profile.action_variants(
            source["mask"],
            seed,
            probability[crop],
            feature,
            allowed_local,
        )

        def score_current(action_tuple):
            action_name, parameters, candidate = action_tuple
            return profile.score_action(
                source,
                action_name,
                parameters,
                candidate,
                probability,
                raw,
                base,
                negative,
                zero_gt,
                zero_gt,
                zero_labels,
                available_sizes,
                feature,
            )

        scored_public = list(score_pool.map(score_current, actions))
        for (action_name, parameters, candidate), public in zip(
            actions, scored_public
        ):
            records.append({"public": public, "crop": crop, "mask": candidate})
'''
if source.count(score_old) != 1:
    raise RuntimeError("v735 action scoring block changed")
source = source.replace(score_old, score_new)
shutdown_old = '''    if not records:
        raise RuntimeError("GT-free runtime candidate family is empty")
'''
shutdown_new = '''    score_pool.shutdown(wait=True)
    if not records:
        raise RuntimeError("GT-free runtime candidate family is empty")
'''
if source.count(shutdown_old) != 1:
    raise RuntimeError("v735 score-pool shutdown anchor changed")
source = source.replace(shutdown_old, shutdown_new)

# Store source/action masks bit-packed between uses.  np.packbits followed by
# np.unpackbits(count=prod(shape)).reshape(shape) is bit-exact for bool masks.
source_pack_old = '''    source_records, seed_summary = profile.generator.construct_object_records(
        probability, raw, base, negative, zero_gt
    )
    seed_mask = (probability >= 0.50) & ~base & ~negative
'''
source_pack_new = '''    source_records, seed_summary = profile.generator.construct_object_records(
        probability, raw, base, negative, zero_gt
    )
    for source_record in source_records:
        source_mask = np.asarray(source_record["mask"], dtype=bool)
        source_record["_packed_mask"] = np.packbits(source_mask, axis=None)
        source_record["_mask_shape"] = tuple(source_mask.shape)
        source_record["mask"] = None
    seed_mask = (probability >= 0.50) & ~base & ~negative
'''
if source.count(source_pack_old) != 1:
    raise RuntimeError("v735 source-mask pack anchor changed")
source = source.replace(source_pack_old, source_pack_new)
unpack_old = '''        crop = source["crop"]
        seed_id = int(source["public"]["seed_id"])
'''
unpack_new = '''        crop = source["crop"]
        source_mask = np.unpackbits(
            source["_packed_mask"],
            count=int(np.prod(source["_mask_shape"])),
        ).reshape(source["_mask_shape"]).astype(bool, copy=False)
        seed_id = int(source["public"]["seed_id"])
'''
if source.count(unpack_old) != 1:
    raise RuntimeError("v735 source-mask unpack anchor changed")
source = source.replace(unpack_old, unpack_new)
if source.count('            source["mask"],\n') != 1:
    raise RuntimeError("v735 action source-mask argument changed")
source = source.replace('            source["mask"],\n', '            source_mask,\n')
record_old = '''            records.append({"public": public, "crop": crop, "mask": candidate})
'''
record_new = '''            records.append({
                "public": public,
                "crop": crop,
                "packed_mask": np.packbits(candidate, axis=None),
                "mask_shape": tuple(candidate.shape),
            })
        source["_packed_mask"] = None
'''
if source.count(record_old) != 1:
    raise RuntimeError("v735 action-mask storage anchor changed")
source = source.replace(record_old, record_new)
union_old = '''def union_mask(shape: tuple[int, ...], selected: list[dict]) -> np.ndarray:
    value = np.zeros(shape, dtype=bool)
    for record in selected:
        value[record["crop"]] |= record["mask"]
    return value
'''
union_new = '''def union_mask(shape: tuple[int, ...], selected: list[dict]) -> np.ndarray:
    value = np.zeros(shape, dtype=bool)
    for record in selected:
        candidate = np.unpackbits(
            record["packed_mask"], count=int(np.prod(record["mask_shape"]))
        ).reshape(record["mask_shape"]).astype(bool, copy=False)
        value[record["crop"]] |= candidate
    return value
'''
if source.count(union_old) != 1:
    raise RuntimeError("v735 packed union anchor changed")
source = source.replace(union_old, union_new)

# Add the cache audit without changing selection features or the output mask.
audit_old = '''        "candidate_action_count": len(records),
        "candidate_seed_count": int(frame["seed_id"].nunique()),
'''
audit_new = '''        "candidate_action_count": len(records),
        "feature_crop_cache_policy": "one feature read per seed; exact candidate preservation",
        "action_scoring_parallelism": 4,
        "action_scoring_order_preserved": True,
        "source_mask_storage": "np.packbits; exact unpack count and shape",
        "action_mask_storage": "np.packbits; selected masks unpacked for union",
        "candidate_seed_count": int(frame["seed_id"].nunique()),
'''
if source.count(audit_old) != 1:
    raise RuntimeError("v718 selection audit block changed")
source = source.replace(audit_old, audit_new)
source = source.replace('"schema_version": 718,', '"schema_version": 735,')

namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(compile(source, str(SOURCE), "exec"), namespace)
