#!/usr/bin/env python3
"""Border-aware source sampler with frozen 485/485 and 850/120 ratios."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name(
    "run_source_balanced_boundary_fallback_v587.py"
)
wrapper_source = SOURCE.read_text(encoding="utf-8")
old_exec = 'exec(compile(source, str(SOURCE), "exec"), globals())\n'
injected_exec = r"""
# ``source`` is now the border-aware transformed v579 program.  Add one
# base-preserving allocation while retaining 1000 total patches and the same
# 30 legacy context centres.
old_policy_gate = '''    if policy != "source_equal_485_485_30":
        raise ValueError(f"Unsupported sampling policy: {policy}")
'''
new_policy_gate = '''    if policy not in {
        "source_equal_485_485_30",
        "source_base85_850_120_30",
    }:
        raise ValueError(f"Unsupported sampling policy: {policy}")
'''
if source.count(old_policy_gate) != 1:
    raise RuntimeError("v579 policy gate changed")
source = source.replace(old_policy_gate, new_policy_gate)

old_quota = '''            negative_quota = int(round(0.03 * num_samples))
            positive_quota = int(num_samples) - negative_quota
            requested_base_quota = positive_quota // 2
            requested_new2_quota = positive_quota - requested_base_quota
            empty_new2_fallback_applied = len(new2_coordinates) == 0
            if empty_new2_fallback_applied:
                # A conservative upstream abstention is a valid causal result,
                # not a sampler error.  There is no new supervision to balance,
                # so retain the fixed 3% context quota and spend the remaining
                # patches on the only available positive stratum.
                base_quota = positive_quota
                new2_quota = 0
            else:
                base_quota = requested_base_quota
                new2_quota = requested_new2_quota
'''
new_quota = '''            negative_quota = int(round(0.03 * num_samples))
            positive_quota = int(num_samples) - negative_quota
            if config["policy"] == "source_equal_485_485_30":
                requested_base_quota, requested_new2_quota = 485, 485
            elif config["policy"] == "source_base85_850_120_30":
                requested_base_quota, requested_new2_quota = 850, 120
            else:
                raise RuntimeError("Unreachable sampling policy")
            if (
                requested_base_quota
                + requested_new2_quota
                + negative_quota
                != int(num_samples)
            ):
                raise RuntimeError("Frozen source quotas do not sum to 1000")
            empty_new2_fallback_applied = len(new2_coordinates) == 0
            if empty_new2_fallback_applied:
                base_quota = positive_quota
                new2_quota = 0
            else:
                base_quota = requested_base_quota
                new2_quota = requested_new2_quota
'''
if source.count(old_quota) != 1:
    raise RuntimeError("v579 quota block changed")
source = source.replace(old_quota, new_quota)
exec(compile(source, str(SOURCE), "exec"), globals())
"""
if wrapper_source.count(old_exec) != 1:
    raise RuntimeError("v587 execution block changed")
wrapper_source = wrapper_source.replace(old_exec, injected_exec)
exec(compile(wrapper_source, str(SOURCE), "exec"), globals())
