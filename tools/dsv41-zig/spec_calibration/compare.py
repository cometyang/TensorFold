"""Compare matching holdout requests and report measured decode speed separately from token correctness."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def key(run):
    return tuple(run.get(field) for field in ("fixture", "prompt_sha256", "seed", "temperature", "max_tokens", "ignore_eos"))


def token_equal(left, right):
    if any(run.get("error") or not run.get("complete") or not isinstance(run.get("token_ids"), list)
           for run in (left, right)):
        return None
    if key(left) != key(right):
        return None
    return left["token_ids"] == right["token_ids"]


def rate_exclusion(batch):
    if batch.get("warmup"):
        return "warmup"
    metrics = batch.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("valid") is not True:
        return "invalid_or_missing_counter_attribution"
    runs = batch.get("runs")
    if not runs or any(run.get("error") or run.get("complete") is not True for run in runs):
        return "incomplete_or_failed_request"
    if any(type(run.get("draft")) is not bool for run in runs):
        return "unknown_draft_mode"
    rate = batch.get("aggregate_decode_tps")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        return "invalid_rate"
    return None


def compare(baseline, candidate, target=0.20):
    reference = defaultdict(list)
    for batch in baseline:
        for run in batch["runs"]:
            reference[key(run)].append(run)
    accuracy, cells = [], defaultdict(lambda: {"baseline": [], "candidate": []})
    for batch in candidate:
        for run in batch["runs"]:
            matches = reference.get(key(run), [])
            valid = [match for match in matches if token_equal(match, run) is not None]
            equal = all(token_equal(match, run) for match in valid) if valid else None
            first_difference = None
            if equal is False:
                other = next(match["token_ids"] for match in valid if token_equal(match, run) is False)
                ids = run["token_ids"]
                first_difference = next((i for i, (a, b) in enumerate(zip(other, ids)) if a != b), min(len(other), len(ids)))
            accuracy.append({"fixture": run["fixture"], "temperature": run["temperature"], "seed": run["seed"],
                             "equal": equal, "first_difference": first_difference,
                             "retrieval": run.get("retrieval"), "split": run.get("split")})
    excluded = {"baseline": Counter(), "candidate": Counter()}
    for label, batches in (("baseline", baseline), ("candidate", candidate)):
        for batch in batches:
            reason = rate_exclusion(batch)
            if reason:
                excluded[label][reason] += 1
                continue
            runs = batch["runs"]
            cell = tuple(sorted(key(run) + (run["draft"],) for run in runs))
            cells[cell][label].append(batch["aggregate_decode_tps"])
    speeds = []
    for cell, samples in cells.items():
        a, b = samples["baseline"], samples["candidate"]
        ma, mb = statistics.median(a) if a else None, statistics.median(b) if b else None
        gain = mb / ma - 1 if ma and mb else None
        speeds.append({"fixtures": [item[0] for item in cell], "temperature": cell[0][3], "streams": len(cell),
                       "draft_modes": [item[-1] for item in cell],
                       "baseline_samples": a, "candidate_samples": b, "baseline_median_tps": ma,
                       "candidate_median_tps": mb, "improvement_fraction": gain,
                       "target_met_observed": gain >= target if gain is not None else None,
                       "enough_repetitions": len(a) >= 3 and len(b) >= 3})
    return {"schema_version": 1, "target_fraction": target, "speed": speeds, "accuracy": accuracy,
            "speed_exclusions": {label: dict(counts) for label, counts in excluded.items()},
            "accuracy_counts": {str(value): sum(row["equal"] is value for row in accuracy)
                                for value in (True, False, None)},
            "caveat": "Observed medians are not a confidence bound; exactness covers these prompts/seeds only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.20)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = compare(read(args.baseline), read(args.candidate), args.target)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy_counts": result["accuracy_counts"], "speed": result["speed"]}, indent=2))


if __name__ == "__main__":
    main()
