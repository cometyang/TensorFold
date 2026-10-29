"""Fit context cost tables from training traces without inventing missing row measurements."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

DEFAULT_RANGES = ((2048, 4096), (65536, 131072), (1047040, 1048576))


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_ranges(ranges):
    previous = -1
    for low, high in sorted(ranges):
        if type(low) is not int or type(high) is not int or low <= previous or high < low:
            raise ValueError("ranges must be disjoint inclusive integer intervals")
        previous = high


def fit(records, ranges=DEFAULT_RANGES, skip_rounds=3, skip_shape=2, min_samples=5, max_drafts=5):
    validate_ranges(ranges)
    if min(skip_rounds, skip_shape) < 0 or min_samples < 1 or not 0 <= max_drafts <= 5:
        raise ValueError("invalid fitter settings")
    samples, buckets, excluded, seen = defaultdict(list), defaultdict(set), Counter(), Counter()
    overheads, previous = defaultdict(list), None
    for record in records:
        if previous and record.get("_source") != previous[0].get("_source"):
            previous = None
        if record.get("event") in ("release", "header", "summary"):
            previous = None
        if record.get("event") != "verify":
            if previous and record.get("event") == "draft" and request_set(record) != request_set(previous[0]):
                previous = None
            continue
        windows = record.get("windows", [])
        if not windows or any(window.get("drafts_enabled") is not True for window in windows):
            excluded["not_draft_enabled"] += 1
            previous = None
            continue
        if any(window.get("first_round") or window.get("round_index", -1) < skip_rounds for window in windows):
            excluded["request_warmup"] += 1
            previous = None
            continue
        if any(window.get("bucket_changed") for window in windows):
            excluded["bucket_transition"] += 1
            previous = None
            continue
        streams, rows, ns = record.get("streams"), record.get("rows"), record.get("verify_ns")
        if type(streams) is not int or streams not in (1, 2) or type(rows) is not int or rows < streams:
            raise ValueError("invalid streams/rows in verify trace")
        if not positive(ns):
            excluded["invalid_timing"] += 1
            previous = None
            continue
        ends = [window.get("proposed_context_end", window["context"] + window["rows"]) for window in windows]
        end = max(ends)
        band = next(((low, high) for low, high in ranges if low <= end <= high), None)
        if band is None:
            excluded["outside_profile_ranges"] += 1
            previous = None
            continue
        pending_end = max(window["context"] + 1 for window in windows)
        if not band[0] <= pending_end <= band[1]:
            excluded["straddles_profile_range"] += 1
            previous = None
            continue
        shape = (record.get("_source", ""), record.get("bucket"), streams, rows)
        seen[shape] += 1
        if seen[shape] <= skip_shape:
            excluded["shape_warmup"] += 1
            previous = None
            continue
        key = (*band, streams, rows)
        samples[key].append(ns / 1e6)
        buckets[key].add(record.get("bucket"))
        group = (*band, streams)
        if previous and previous[1] == group and request_set(previous[0]) == request_set(record):
            last = previous[0]
            if positive(last.get("start_ns")) and positive(record.get("start_ns")):
                residual = record["start_ns"] - last["start_ns"] - last["verify_ns"]
                if residual >= 0:
                    overheads[group].append(residual / 1e6)
                else:
                    excluded["negative_round_residual"] += 1
        previous = (record, group)
    profiles, coverage = [], []
    for low, high in sorted(ranges):
        for streams in (1, 2):
            required = range(streams, streams * (max_drafts + 1) + 1)
            if not any((low, high, streams, row) in samples for row in required):
                continue
            rows, missing = {}, []
            for row in required:
                key = (low, high, streams, row)
                values = samples[key]
                if len(values) < min_samples:
                    missing.append(row)
                median = statistics.median(values) if values else None
                rows[str(row)] = {"samples": len(values), "median_ms": median,
                                  "mad_ms": statistics.median(abs(v - median) for v in values) if values else None,
                                  "min_ms": min(values) if values else None, "max_ms": max(values) if values else None,
                                  "buckets": sorted(b for b in buckets[key] if b is not None)}
            overhead_values = overheads[(low, high, streams)]
            overhead = statistics.median(overhead_values) if overhead_values else None
            complete = not missing and len(overhead_values) >= min_samples
            coverage.append({"min_context": low, "max_context": high, "streams": streams,
                             "complete": complete, "missing_rows": missing, "rows": rows,
                             "overhead_samples": len(overhead_values), "overhead_median_ms": overhead,
                             "overhead_mad_ms": statistics.median(abs(v - overhead) for v in overhead_values)
                             if overhead_values else None})
            if complete:
                costs = [rows[str(row)]["median_ms"] for row in required]
                # Rows below active stream count cannot be selected; repeat the first measured cost for schema density.
                dense = [costs[0]] * (streams - 1) + costs
                profiles.append({"min_context": low, "max_context": high, "streams": streams,
                                 "forward_ms": dense, "overhead_ms": overhead})
    return {"schema_version": 1, "family": "dsv41", "profiles": profiles,
            "provenance": {"method": "median measured verify_ns / 1e6; no interpolation or extrapolation",
                           "skip_rounds": skip_rounds, "skip_shape_occurrences_per_trace": skip_shape,
                           "min_samples_per_row": min_samples, "max_drafts": max_drafts,
                           "overhead": "median(next verify.start_ns - verify.start_ns - verify.verify_ns) / 1e6",
                           "overhead_filter": "adjacent retained rounds, same request set, source and context range",
                           "unused_rows_below_streams": "filled with first measurable row cost; never selectable",
                           "excluded": dict(excluded), "coverage": coverage}}


def request_set(record):
    return tuple(sorted((window.get("trace_id"), window.get("slot", 0)) for window in record.get("windows", [])))


def load_traces(paths):
    records, sources = [], []
    for path in paths:
        raw = path.read_bytes()
        source = hashlib.sha256(raw).hexdigest()
        sources.append({"name": path.name, "sha256": source})
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != 1:
                raise ValueError("unsupported trace schema")
            row["_source"] = source
            records.append(row)
    return records, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranges", default="2048:4096,65536:131072,1047040:1048576")
    parser.add_argument("--skip-rounds", type=int, default=3)
    parser.add_argument("--skip-shape", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--max-drafts", type=int, default=5)
    args = parser.parse_args()
    ranges = [tuple(map(int, interval.split(":"))) for interval in args.ranges.split(",")]
    records, sources = load_traces(args.traces)
    profile = fit(records, ranges, args.skip_rounds, args.skip_shape, args.min_samples, args.max_drafts)
    profile["provenance"]["sources"] = sources
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles": len(profile["profiles"]), "output": args.output.name,
                      "incomplete": [row for row in profile["provenance"]["coverage"] if not row["complete"]]}))
    if not profile["profiles"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
