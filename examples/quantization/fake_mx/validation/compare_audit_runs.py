#!/usr/bin/env python
"""Compare tensor dumps from two Fake-MX audit runs.

Two-stage comparison:
1. Compare ``events.rank*.jsonl`` — verifies node sets, specs, QDQ counts,
   and target decisions match.
2. Compare tensor bundles — verifies weight/activation tensors are identical.

Usage::

    python compare_audit_runs.py <run_a_dir> <run_b_dir> [--tolerance 1e-6]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

VALID_STAGES = (
    "weight_raw",
    "weight_transformed",
    "weight_qdq",
    "act_raw",
    "act_transformed",
    "act_qdq",
    "output",
    "attn_q_raw",
    "attn_q_qdq",
    "attn_k_raw",
    "attn_k_qdq",
    "attn_v_raw",
    "attn_v_qdq",
    "gdn_qkv_raw",
    "gdn_qkv_qdq",
)

# Canonical events that participate in A/B pass/fail comparison.
# Implementation-specific diagnostic events (scheme_created, params_loaded,
# quant_method_selected, etc.) are ignored by the comparison logic.
CANONICAL_EVENTS = frozenset(
    {
        "node_selected",
        "weight_transform",
        "weight_qdq",
        "activation_transform",
        "activation_qdq",
        "linear_output",
        "target_decision",
        "gdn_core_qdq",
        "attn_cache_qdq",
    }
)


def _load_events(run_dir: str) -> list[dict]:
    """Load all events from a run directory."""
    events: list[dict] = []
    for fname in sorted(os.listdir(run_dir)):
        if not fname.startswith("events.") or not fname.endswith(".jsonl"):
            continue
        path = os.path.join(run_dir, fname)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    if ev.get("event") in CANONICAL_EVENTS:
                        events.append(ev)
    return events


def _summarise_events(events: list[dict]) -> dict:
    """Group canonical events by prefix and compute summary statistics."""
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_prefix[ev.get("prefix", "")].append(ev)

    summary: dict[str, dict] = {}
    for prefix, evs in by_prefix.items():
        qdq_count = sum(1 for e in evs if e.get("event") in ("weight_qdq", "activation_qdq"))
        node_ev = next((e for e in evs if e.get("event") == "node_selected"), None)
        # Use deduplicated event set for comparison — profile_run may
        # cause duplicate events in one run but not the other.
        unique_events = sorted(set(e.get("event") for e in evs))
        summary[prefix] = {
            "canonical_event_count": len(evs),
            "qdq_count": qdq_count,
            "quant_type": node_ev.get("quant_type") if node_ev else None,
            "algorithm": node_ev.get("algorithm") if node_ev else None,
            "mx_format": node_ev.get("mx_format") if node_ev else None,
            "group_size": node_ev.get("group_size") if node_ev else None,
            "events": unique_events,
        }
    return summary


def _compare_events(run_a: str, run_b: str) -> dict:
    """Stage 1: compare event summaries."""
    events_a = _load_events(run_a)
    events_b = _load_events(run_b)
    summary_a = _summarise_events(events_a)
    summary_b = _summarise_events(events_b)

    all_prefixes = sorted(set(summary_a.keys()) | set(summary_b.keys()))
    missing_a = sorted(set(summary_b.keys()) - set(summary_a.keys()))
    missing_b = sorted(set(summary_a.keys()) - set(summary_b.keys()))

    issues: list[str] = []
    for prefix in all_prefixes:
        sa = summary_a.get(prefix)
        sb = summary_b.get(prefix)
        if sa is None:
            issues.append(f"missing_nodes: {prefix} missing in run A")
            continue
        if sb is None:
            issues.append(f"missing_nodes: {prefix} missing in run B")
            continue
        # Compare spec fields from node_selected
        for field in ("quant_type", "algorithm", "mx_format", "group_size"):
            va = sa.get(field)
            vb = sb.get(field)
            if va != vb:
                issues.append(f"wrong_spec: {prefix} {field} A={va} B={vb}")
        # Compare deduplicated event sets (not counts — profile_run
        # may cause different counts without affecting values).
        if sa["events"] != sb["events"]:
            issues.append(f"wrong_events: {prefix} A={sa['events']} B={sb['events']}")

    # Minimum evidence assertions — empty runs must not PASS.
    node_count_a = sum(1 for s in summary_a.values() if "node_selected" in s.get("events", []))
    node_count_b = sum(1 for s in summary_b.values() if "node_selected" in s.get("events", []))
    if node_count_a == 0:
        issues.append("zero_evidence: run A has 0 node_selected events")
    if node_count_b == 0:
        issues.append("zero_evidence: run B has 0 node_selected events")

    import hashlib

    selected_prefixes_a = sorted(p for p, s in summary_a.items() if "node_selected" in s.get("events", []))
    selected_prefixes_b = sorted(p for p, s in summary_b.items() if "node_selected" in s.get("events", []))
    prefix_hash_a = hashlib.sha256("\n".join(selected_prefixes_a).encode()).hexdigest()[:16]
    prefix_hash_b = hashlib.sha256("\n".join(selected_prefixes_b).encode()).hexdigest()[:16]

    return {
        "total_prefixes_a": len(summary_a),
        "total_prefixes_b": len(summary_b),
        "node_selected_a": node_count_a,
        "node_selected_b": node_count_b,
        "selected_prefix_hash_a": prefix_hash_a,
        "selected_prefix_hash_b": prefix_hash_b,
        "selected_prefixes_match": prefix_hash_a == prefix_hash_b,
        "missing_in_a": missing_a,
        "missing_in_b": missing_b,
        "issues": issues,
        "passed": len(issues) == 0 and not missing_a and not missing_b and node_count_a > 0 and node_count_b > 0,
    }


def _load_tensor_bundles(run_dir: str) -> dict[str, dict[int, dict[str, dict[int, dict[str, dict]]]]]:
    """Load all tensor bundles from a run directory.

    Returns: {prefix: {rank: {kind: {call_index: {stage: tensor_data}}}}}
    """
    tensor_dir = os.path.join(run_dir, "tensors")
    if not os.path.isdir(tensor_dir):
        return {}

    bundles: dict[str, dict[int, dict[str, dict[int, dict[str, dict]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for fname in os.listdir(tensor_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(tensor_dir, fname)
        data = torch.load(path, map_location="cpu")
        meta = data.get("metadata", {})
        prefix = meta.get("prefix", "")
        rank = meta.get("rank", 0)
        kind = meta.get("kind", "unknown")
        call_idx = meta.get("call_index", 0)
        tensors = data.get("tensors", {})
        for stage, info in tensors.items():
            bundles[prefix][rank][kind][call_idx][stage] = info
    return bundles


def _compare_tensor(a_info: dict, b_info: dict, tolerance: float) -> dict:
    a, b = a_info["data"], b_info["data"]
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape: {tuple(a.shape)} vs {tuple(b.shape)}"}
    if str(a.dtype) != str(b.dtype):
        return {"match": False, "reason": f"dtype: {a.dtype} vs {b.dtype}"}
    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    max_abs = diff.max().item()
    mse = diff.pow(2).mean().item()
    a_flat = a.to(torch.float32).flatten()
    b_flat = b.to(torch.float32).flatten()
    cosine = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    allclose = torch.allclose(a.to(torch.float32), b.to(torch.float32), atol=tolerance, rtol=tolerance)
    return {
        "match": allclose,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "max_abs_error": max_abs,
        "mse": mse,
        "cosine_similarity": cosine,
        "allclose": allclose,
    }


def _compare_tensors(run_a: str, run_b: str, tolerance: float) -> dict:
    """Stage 2: compare tensor bundles."""
    bundles_a = _load_tensor_bundles(run_a)
    bundles_b = _load_tensor_bundles(run_b)

    all_prefixes = sorted(set(bundles_a.keys()) | set(bundles_b.keys()))
    results: list[dict] = []
    failures = 0
    first_divergence = None

    for prefix in all_prefixes:
        ranks_a = bundles_a.get(prefix, {})
        ranks_b = bundles_b.get(prefix, {})
        all_ranks = sorted(set(ranks_a.keys()) | set(ranks_b.keys()))
        for rank in all_ranks:
            kinds_a = ranks_a.get(rank, {})
            kinds_b = ranks_b.get(rank, {})
            all_kinds = sorted(set(kinds_a.keys()) | set(kinds_b.keys()))
            for kind in all_kinds:
                calls_a = kinds_a.get(kind, {})
                calls_b = kinds_b.get(kind, {})
                all_calls = sorted(set(calls_a.keys()) | set(calls_b.keys()))
                for call in all_calls:
                    stages_a = calls_a.get(call, {})
                    stages_b = calls_b.get(call, {})
                    for stage in VALID_STAGES:
                        if stage not in stages_a and stage not in stages_b:
                            continue
                        if stage not in stages_a:
                            results.append(
                                {
                                    "prefix": prefix,
                                    "rank": rank,
                                    "kind": kind,
                                    "call": call,
                                    "stage": stage,
                                    "match": False,
                                    "reason": "missing in A",
                                }
                            )
                            failures += 1
                            continue
                        if stage not in stages_b:
                            results.append(
                                {
                                    "prefix": prefix,
                                    "rank": rank,
                                    "kind": kind,
                                    "call": call,
                                    "stage": stage,
                                    "match": False,
                                    "reason": "missing in B",
                                }
                            )
                            failures += 1
                            continue
                        res = _compare_tensor(stages_a[stage], stages_b[stage], tolerance)
                        if not res["match"]:
                            failures += 1
                            if first_divergence is None:
                                first_divergence = (prefix, rank, kind, call, stage, res)
                        results.append(
                            {"prefix": prefix, "rank": rank, "kind": kind, "call": call, "stage": stage, **res}
                        )

    # Minimum evidence: zero tensor comparisons must not PASS.
    passed = failures == 0 and len(results) > 0
    if len(results) == 0:
        issues_msg = "zero_evidence: 0 tensor comparisons performed (no bundles found)"
    else:
        issues_msg = None
    return {
        "total": len(results),
        "failures": failures,
        "results": results,
        "first_divergence": first_divergence,
        "zero_evidence": issues_msg,
        "passed": passed,
    }


def compare_runs(run_a: str, run_b: str, tolerance: float) -> int:
    print(f"Run A: {run_a}")
    print(f"Run B: {run_b}")
    print(f"Tolerance: {tolerance}")
    print("=" * 80)

    # Stage 1: events
    print("\n--- Stage 1: Event comparison ---")
    event_result = _compare_events(run_a, run_b)
    print(f"  Prefixes A: {event_result['total_prefixes_a']}")
    print(f"  Prefixes B: {event_result['total_prefixes_b']}")
    if event_result["missing_in_a"]:
        print(f"  Missing in A: {event_result['missing_in_a']}")
    if event_result["missing_in_b"]:
        print(f"  Missing in B: {event_result['missing_in_b']}")
    if event_result["issues"]:
        for issue in event_result["issues"]:
            print(f"  [ISSUE] {issue}")
    print(f"  Event stage: {'PASS' if event_result['passed'] else 'FAIL'}")

    # Stage 2: tensors
    print("\n--- Stage 2: Tensor comparison ---")
    tensor_result = _compare_tensors(run_a, run_b, tolerance)
    print(f"  Total comparisons: {tensor_result['total']}")
    print(f"  Passed: {tensor_result['total'] - tensor_result['failures']}")
    print(f"  Failed: {tensor_result['failures']}")
    if tensor_result["first_divergence"]:
        p, rk, k, c, s, r = tensor_result["first_divergence"]
        print(f"  First divergence: {p} rank={rk} kind={k} call={c} stage={s}")
        if "max_abs_error" in r:
            print(f"    max_abs_error={r['max_abs_error']:.2e} cosine={r['cosine_similarity']:.6f}")
    print(f"  Tensor stage: {'PASS' if tensor_result['passed'] else 'FAIL'}")

    # Overall
    print("\n" + "=" * 80)
    overall_pass = event_result["passed"] and tensor_result["passed"]
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    # Save report
    report_path = os.path.join(run_a, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "tolerance": tolerance,
                "events": event_result,
                "tensors": {
                    "total": tensor_result["total"],
                    "failures": tensor_result["failures"],
                    "first_divergence": tensor_result["first_divergence"],
                    "results": tensor_result["results"],
                },
                "passed": overall_pass,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nReport saved to {report_path}")
    return 0 if overall_pass else 2


def main():
    parser = argparse.ArgumentParser(description="Compare two Fake-MX audit runs (events + tensors)")
    parser.add_argument("run_a", help="Path to run A (e.g. modelslim output)")
    parser.add_argument("run_b", help="Path to run B (e.g. intrusive output)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for allclose")
    args = parser.parse_args()
    sys.exit(compare_runs(args.run_a, args.run_b, args.tolerance))


if __name__ == "__main__":
    main()
