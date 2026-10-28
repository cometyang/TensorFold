"""Summarize measured speculative traces; cost-policy replay predicts counterfactual choices, not actual TPS."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import fit

DEFAULT_FORWARD_MS = (30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9,
                      75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2)
COMPONENTS = ("verify_ns", "send_ns", "readback_ns", "sample_ns", "total_ns")


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def distribution(values):
    if not values:
        return {"samples": 0, "median": None, "p10": None, "p90": None, "sum": 0}
    ordered = sorted(values)

    def percentile(q):
        at = (len(ordered) - 1) * q
        low = int(at)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (at - low)

    return {"samples": len(values), "median": statistics.median(values), "p10": percentile(0.1),
            "p90": percentile(0.9), "sum": sum(values)}


def survivals(logits):
    if not isinstance(logits, list) or any(not number(value, -math.inf) for value in logits):
        return None
    out, reach = [], 1.0
    for value in logits:
        if value >= 0:
            probability = 1 / (1 + math.exp(-value))
        else:
            exp = math.exp(value)
            probability = exp / (1 + exp)
        reach *= probability
        out.append(reach)
    return out


def evaluate(probabilities, counts, costs, overhead):
    rows = len(probabilities) + sum(counts)
    expected = len(probabilities) + sum(sum(p[:count]) for p, count in zip(probabilities, counts))
    cost = costs[rows - 1] + overhead
    return {"counts": list(counts), "rows": rows, "expected_tokens": expected,
            "cost_ms": cost, "expected_tokens_per_ms": expected / cost}


def choose(probabilities, costs, overhead, max_rows=16):
    if not probabilities or max_rows < len(probabilities):
        raise ValueError("invalid stream count or row limit")
    possible = min(max_rows, len(probabilities) + sum(map(len, probabilities)))
    if len(costs) < possible or any(not number(cost) or cost == 0 for cost in costs):
        raise ValueError("complete finite positive cost coverage is required")
    if not number(overhead):
        raise ValueError("invalid overhead")
    for probs in probabilities:
        if any(not number(p) or p > 1 for p in probs) or any(a < b for a, b in zip(probs, probs[1:])):
            raise ValueError("prefix survival probabilities must be finite, bounded and nonincreasing")
    counts = [0] * len(probabilities)
    best = evaluate(probabilities, counts, costs, overhead)
    for _ in range(possible - len(probabilities)):
        available = [i for i, p in enumerate(probabilities) if counts[i] < len(p)]
        pick = max(available, key=lambda i: (probabilities[i][counts[i]], -i))
        counts[pick] += 1
        candidate = evaluate(probabilities, counts, costs, overhead)
        if candidate["expected_tokens_per_ms"] > best["expected_tokens_per_ms"]:
            best = candidate
    return best


def profiles_from(catalog):
    if catalog.get("schema_version") != 1 or catalog.get("family") != "dsv41":
        raise ValueError("expected a schema_version 1 dsv41 catalog")
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("profiles must be an array")
    out = []
    for original in profiles:
        profile = dict(original, min_context=original.get("min_context", 0))
        low, high, streams = (profile.get(key) for key in ("min_context", "max_context", "streams"))
        costs = profile.get("forward_ms")
        if (not integer(low) or not integer(high, 1) or low > high or streams not in (1, 2)
                or type(streams) is not int or not isinstance(costs, list) or len(costs) < streams
                or any(not number(cost) or cost == 0 for cost in costs)):
            raise ValueError("invalid context cost profile")
        overhead = profile.get("overhead_ms")
        if overhead is not None and not number(overhead):
            raise ValueError("invalid measured overhead")
        out.append(profile)
    out.sort(key=lambda p: (p["streams"], p["max_context"]))
    for left, right in zip(out, out[1:]):
        if left["streams"] == right["streams"] and right["min_context"] <= left["max_context"]:
            raise ValueError("overlapping inclusive context cost profiles")
    return out


def lane_key(source, window):
    return source, window.get("trace_id"), window.get("slot", 0)


def observed_shape(record):
    windows = record.get("windows")
    streams, rows = record.get("streams"), record.get("rows")
    if type(streams) is not int or streams not in (1, 2) or not isinstance(windows, list) or len(windows) != streams:
        return None
    if not integer(rows, streams) or any(not integer(w.get("context")) or not integer(w.get("rows"), 1)
                                        or not integer(w.get("trace_id"), 1) for w in windows):
        return None
    if sum(w["rows"] for w in windows) != rows:
        return None
    return max(w["context"] + 1 for w in windows), max(w["context"] + w["rows"] for w in windows)


def new_group():
    return {"components": defaultdict(list), "rows": Counter(), "selected_k": Counter(), "acceptance": {},
            "calibration": defaultdict(list), "bins": defaultdict(list), "policy": [], "excluded": Counter(),
            "verify_events": 0, "sweep_events": 0, "context_min": None, "context_max": None}


def acceptance_stats(group, windows):
    for window in windows:
        selected = window.get("selected_drafts")
        accepted = window.get("potential_accepted_drafts")
        if not integer(selected) or selected != window["rows"] - 1 or not integer(accepted) or accepted > selected:
            group["excluded"]["invalid_acceptance"] += 1
            continue
        group["selected_k"][selected] += 1
        cell = group["acceptance"].setdefault(selected, {"accepted": [], "predicted": []})
        cell["accepted"].append(accepted)
        logits = window.get("confidence_logits")
        probs = survivals(logits)
        if window.get("host_drafts", 0) != 0 or probs is None or len(probs) != selected:
            group["excluded"]["incomplete_selected_confidences"] += 1
            continue
        cell["predicted"].append(sum(probs))
        for depth, probability in enumerate(probs, 1):
            observed = int(accepted >= depth)
            group["calibration"][depth].append((probability, observed))
            group["bins"][(depth, min(9, int(probability * 10)))].append((probability, observed))


def paired_drafts(record, matches):
    if any(match is None or match[0].get("context") != window["context"]
           for window, match in zip(record["windows"], matches)):
        return None, "missing_or_stale_preceding_draft"
    probabilities = []
    for window, match in zip(record["windows"], matches):
        draft_window = match[0]
        probs = survivals(draft_window.get("confidence_logits"))
        if (window.get("drafts_enabled") is not True or window.get("host_drafts", 0) != 0
                or probs is None or not probs or len(probs) != draft_window.get("held_drafts")
                or len(probs) != draft_window.get("rows") or len(probs) < window.get("held_drafts", 0)
                or window.get("selected_drafts") != window.get("held_drafts")
                or window.get("selected_drafts") != window["rows"] - 1):
            return None, "incomplete_full_held_confidences"
        probabilities.append(probs)
    return probabilities, None


def draft_components(group, windows, matches, source):
    if any(match is None or match[0].get("context") != window["context"]
           for window, match in zip(windows, matches)):
        return
    drafts = {match[1]: match[2] for match in matches}
    wanted = {lane_key(source, window) for window in windows}
    drafted = {lane_key(source, window) for draft in drafts.values() for window in draft["windows"]}
    if drafted != wanted:
        group["excluded"]["draft_timing_mixed_request_set"] += 1
        return
    for field in ("pass_ns", "total_ns"):
        values = [draft.get(field) for draft in drafts.values()]
        if all(number(value) for value in values):
            group["components"]["preceding_draft_" + field.removesuffix("_ns")].append(sum(values) / 1e6)


def policy_replay(record, probabilities, profiles, legacy_costs, legacy_overhead, max_rows):
    windows = record["windows"]
    start = max(w["context"] + 1 for w in windows)
    end = max(w["context"] + 1 + len(p) for w, p in zip(windows, probabilities))
    profile = next((p for p in profiles if p["streams"] == len(windows)
                    and p["min_context"] <= start <= end <= p["max_context"]), None)
    if profile is None:
        return None, "no_profile_for_full_proposal_span"
    possible = min(max_rows, len(windows) + sum(map(len, probabilities)))
    if min(len(profile["forward_ms"]), len(legacy_costs)) < possible:
        return None, "incomplete_feasible_row_costs"
    overhead = profile.get("overhead_ms")
    if overhead is None:
        return None, "missing_measured_overhead"
    legacy = choose(probabilities, legacy_costs, legacy_overhead, max_rows)
    measured = choose(probabilities, profile["forward_ms"], overhead, max_rows)
    legacy_at_measured = evaluate(probabilities, legacy["counts"], profile["forward_ms"], overhead)
    gain = measured["expected_tokens_per_ms"] / legacy_at_measured["expected_tokens_per_ms"] - 1
    observed = [window["selected_drafts"] for window in windows]
    return {"legacy": legacy_at_measured, "measured": measured, "gain": gain,
            "changed": legacy["counts"] != measured["counts"],
            "legacy_matches_observed": legacy["counts"] == observed,
            "forced_sweep": record.get("calibration_k") is not None}, None


def calibration_cell(values):
    predicted = statistics.mean(p for p, _ in values)
    actual = statistics.mean(y for _, y in values)
    return {"samples": len(values), "predicted_prefix_survival": predicted, "observed_prefix_survival": actual,
            "observed_minus_predicted": actual - predicted, "brier_score": statistics.mean((p-y)**2 for p, y in values)}


def finish_group(key, group, target):
    low, high, streams, greedy, bucket = key
    policies = group["policy"]
    policy = {"label": "predicted_counterfactual_not_actual_tps", "eligible_rounds": len(policies),
              "changed_choices": sum(p["changed"] for p in policies),
              "round_gain_fraction": distribution([p["gain"] for p in policies]),
              "rounds_modeling_target": sum(p["gain"] >= target for p in policies),
              "legacy_selected_rows": dict(Counter(p["legacy"]["rows"] for p in policies)),
              "measured_selected_rows": dict(Counter(p["measured"]["rows"] for p in policies))}
    policy["changed_fraction"] = policy["changed_choices"] / len(policies) if policies else None
    if policies:
        rates = {name: sum(p[name]["expected_tokens"] for p in policies) / sum(p[name]["cost_ms"] for p in policies)
                 for name in ("legacy", "measured")}
        policy["modeled_gain_on_replayed_states"] = rates["measured"] / rates["legacy"] - 1
    else:
        policy["modeled_gain_on_replayed_states"] = None
    natural = [p for p in policies if not p["forced_sweep"]]
    policy["legacy_scenario_matches_observed_fraction"] = (sum(p["legacy_matches_observed"] for p in natural)
                                                           / len(natural) if natural else None)
    accepted = [{"verified_k": k, "windows": len(cell["accepted"]),
                 "potential_accepted_histogram": dict(Counter(cell["accepted"])),
                 "mean_potential_accepted_drafts": statistics.mean(cell["accepted"]),
                 "mean_potential_tokens_including_pending": 1 + statistics.mean(cell["accepted"]),
                 "confidence_windows": len(cell["predicted"]),
                 "mean_predicted_accepted_drafts": statistics.mean(cell["predicted"]) if cell["predicted"] else None}
                for k, cell in sorted(group["acceptance"].items())]
    return {"min_context": low, "max_context": high, "streams": streams, "greedy": greedy, "bucket": bucket,
            "observed_pending_context_min": group["context_min"], "observed_proposed_context_max": group["context_max"],
            "verify_events": group["verify_events"], "sweep_events": group["sweep_events"],
            "component_ms": {name: distribution(values) for name, values in sorted(group["components"].items())},
            "verified_rows_histogram": dict(sorted(group["rows"].items())),
            "selected_k_histogram": dict(sorted(group["selected_k"].items())),
            "potential_acceptance_by_verified_k": accepted,
            "prefix_calibration": [{"depth": depth, **calibration_cell(values)}
                                   for depth, values in sorted(group["calibration"].items())],
            "prefix_calibration_bins": [{"depth": depth, "probability_low": b / 10, "probability_high": (b+1) / 10,
                                        **calibration_cell(values)} for (depth, b), values in sorted(group["bins"].items())],
            "policy": policy, "excluded": dict(group["excluded"])}


def analyze(records, catalog, legacy_costs=DEFAULT_FORWARD_MS, legacy_overhead_ms=8.0,
            max_rows=16, skip_rounds=3, skip_shape=2, target=0.20):
    profiles = profiles_from(catalog)
    if (not integer(max_rows, 1) or not integer(skip_rounds) or not integer(skip_shape)
            or not number(legacy_overhead_ms) or not number(target) or not legacy_costs
            or any(not number(cost) or cost == 0 for cost in legacy_costs)):
        raise ValueError("invalid analysis options")
    ranges = {(p["min_context"], p["max_context"]) for p in profiles}
    ranges.update((p["min_context"], p["max_context"]) for p in catalog.get("provenance", {}).get("coverage", []))
    ranges = sorted(ranges or fit.DEFAULT_RANGES)
    groups, pending, seen, excluded, completeness = {}, {}, Counter(), Counter(), {}
    for index, record in enumerate(records):
        source, event = record.get("_source", "memory"), record.get("event")
        completeness.setdefault(source, False)
        if event in ("header", "summary"):
            pending = {key: value for key, value in pending.items() if key[0] != source}
            completeness[source] = event == "summary" and record.get("complete") is True
            continue
        completeness[source] = False
        windows = record.get("windows", [])
        if event == "release":
            for window in windows:
                pending.pop(lane_key(source, window), None)
            continue
        if event == "draft":
            for window in windows:
                pending[lane_key(source, window)] = window, index, record
            continue
        if event != "verify":
            continue
        matches = [pending.pop(lane_key(source, window), None) for window in windows]
        shape = observed_shape(record)
        if shape is None:
            excluded["invalid_verify_shape"] += 1
            continue
        if any(w.get("first_round") or w.get("round_index", -1) < skip_rounds for w in windows):
            excluded["request_warmup"] += 1
            continue
        if any(w.get("bucket_changed") for w in windows):
            excluded["bucket_transition"] += 1
            continue
        start, end = shape
        streams, bucket, greedy = record["streams"], record.get("bucket"), record.get("greedy")
        preferred = [(p["min_context"], p["max_context"]) for p in profiles if p["streams"] == streams]
        band = next(((low, high) for low, high in preferred + ranges if low <= start <= end <= high), (None, None))
        if type(greedy) is not bool or not integer(bucket):
            excluded["missing_mode_or_bucket"] += 1
            continue
        shape_key = source, bucket, streams, record["rows"], greedy
        seen[shape_key] += 1
        if seen[shape_key] <= skip_shape:
            excluded["shape_warmup"] += 1
            continue
        key = *band, streams, greedy, bucket
        group = groups.setdefault(key, new_group())
        group["verify_events"] += 1
        group["sweep_events"] += int(record.get("calibration_k") is not None)
        group["context_min"] = min(group["context_min"] if group["context_min"] is not None else start, start)
        group["context_max"] = max(group["context_max"] or end, end)
        group["rows"][record["rows"]] += 1
        for name in COMPONENTS:
            value = record.get(name)
            if number(value):
                group["components"][name.removesuffix("_ns")].append(value / 1e6)
            else:
                group["excluded"]["invalid_or_missing_" + name] += 1
        acceptance_stats(group, windows)
        draft_components(group, windows, matches, source)
        probs, reason = paired_drafts(record, matches)
        if reason:
            group["excluded"][reason] += 1
            continue
        if any(not integer(w.get("selected_drafts")) for w in windows):
            group["excluded"]["invalid_policy_selected_depth"] += 1
            continue
        replay, reason = policy_replay(record, probs, profiles, legacy_costs, legacy_overhead_ms, max_rows)
        if reason:
            group["excluded"][reason] += 1
        else:
            group["policy"].append(replay)
    ordered = sorted(groups.items(), key=lambda item: tuple(-1 if v is None else v for v in item[0]))
    return {"schema_version": 1, "target_fraction": target,
            "groups": [finish_group(key, group, target) for key, group in ordered], "excluded": dict(excluded),
            "trace_complete_by_source": completeness,
            "settings": {"skip_rounds": skip_rounds, "skip_shape": skip_shape, "max_rows": max_rows,
                         "legacy_forward_ms": list(legacy_costs), "legacy_fixed_overhead_ms": legacy_overhead_ms},
            "limitations": ["CPU-wall component durations include existing synchronization; these are not isolated GPU kernels.",
                            "Policy replay is a confidence-based counterfactual on observed states, not measured TPS or a forecast.",
                            "The fixed legacy overhead is an explicit scenario; runtime learned overhead is not recorded exactly.",
                            "Only preceding full held vectors are replayed; unverified acceptance depths are never inferred.",
                            "Potential acceptance precedes stop and output-budget handling; it is not committed token throughput.",
                            "Confidence calibration uses observed selected depths and can carry selection bias.",
                            "Overhead can depend on sampling mode and draft depth; replay requires the catalog's fitting conditions.",
                            "Training replay does not replace independent holdout speed and exact-token benchmarks."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-round-ms", help="actual TF_DS_ROUND_MS, if overridden; otherwise the source default")
    parser.add_argument("--legacy-overhead-ms", type=float, default=8.0)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--skip-rounds", type=int, default=3)
    parser.add_argument("--skip-shape", type=int, default=2)
    parser.add_argument("--target", type=float, default=0.20)
    args = parser.parse_args()
    records, sources = fit.load_traces(args.traces)
    costs = [float(value) for value in args.legacy_round_ms.split(",")] if args.legacy_round_ms else DEFAULT_FORWARD_MS
    result = analyze(records, json.loads(args.catalog.read_text()), costs, args.legacy_overhead_ms,
                     args.max_rows, args.skip_rounds, args.skip_shape, args.target)
    result["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(result["groups"]), "output": str(args.output),
                      "prediction_only": True, "trace_complete": all(result["trace_complete_by_source"].values())}))


if __name__ == "__main__":
    main()
