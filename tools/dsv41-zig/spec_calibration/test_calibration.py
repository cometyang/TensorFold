"""Regression checks for incomplete measurements and misleading correctness/speed comparisons."""

import io
import hashlib
import json
import unittest
from unittest.mock import patch

import benchmark
import compare
import fit
import fixtures


def event(rows, index, ns=10000000, context=2048, streams=1):
    return {"schema_version": 1, "event": "verify", "rows": rows, "streams": streams,
            "verify_ns": ns, "start_ns": 1000000000 + index * 20000000, "bucket": 4096, "windows": [
                {"trace_id": 1, "context": context, "rows": rows, "round_index": index,
                 "first_round": index == 0, "drafts_enabled": True}]}


def result(ids, **kwargs):
    return {"fixture": "test", "prompt_sha256": "test", "seed": 1, "temperature": 0,
            "max_tokens": 10, "ignore_eos": False, "complete": True, "token_ids": ids, "draft": True, **kwargs}


def batch(rate, draft=True, **kwargs):
    return {"runs": [result([1], draft=draft)], "aggregate_decode_tps": rate,
            "metrics": {"valid": True}, "warmup": False, **kwargs}


class Measurements(unittest.TestCase):
    def test_missing_or_incomplete_tokens_never_pass_exactness(self):
        valid = result([1, 2])
        self.assertTrue(compare.token_equal(valid, result([1, 2])))
        self.assertFalse(compare.token_equal(valid, result([1, 3])))
        for bad in (result(None), result([1, 2], complete=False), result([1, 2], error="failed"),
                    result([1, 2], seed=2)):
            self.assertIsNone(compare.token_equal(valid, bad))

    def test_metrics_reject_resets_or_foreign_requests(self):
        before = {name: 10 for name in benchmark.COUNTERS}
        after = {name: 15 for name in benchmark.COUNTERS}
        self.assertTrue(benchmark.metric_delta(before, after, 5)["valid"])
        self.assertEqual(benchmark.metric_delta(before, after, 4)["error"], "unattributed_generation")
        after["mtp_accepted_total"] = 0
        self.assertEqual(benchmark.metric_delta(before, after, 5)["error"], "counter_reset")

    def test_stream_validates_usage_against_full_ids(self):
        fixture = {"name": "test", "split": "holdout", "kind": "prose", "prompt": [10, 11],
                   "prompt_sha256": benchmark.digest_ids([10, 11])}
        final = {"usage": {"completion_tokens": 2, "prompt_tokens": 2},
                 "tensorfold": {"token_ids": [4, 5], "token_sha": hashlib.sha256(b"4,5").hexdigest()[:12]}}
        raw = ("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode()
        with patch.object(benchmark, "request", return_value=io.BytesIO(raw)):
            run = benchmark.stream("http://localhost", "test", fixture)
        self.assertEqual(run["token_ids"], [4, 5])
        self.assertNotIn("error", run)
        self.assertNotEqual(run["native_token_sha"], run["token_sha256"][:12])
        final["tensorfold"]["token_ids"] = [4]
        raw = ("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode()
        with patch.object(benchmark, "request", return_value=io.BytesIO(raw)):
            run = benchmark.stream("http://localhost", "test", fixture)
        self.assertEqual(run["error"], "missing_or_inconsistent_token_ids")

    def test_digest_rejects_boolean_ids(self):
        with self.assertRaises(ValueError):
            benchmark.digest_ids([True])

    def test_warmups_do_not_make_twenty_percent_claim(self):
        baseline = [batch(100)]
        candidate = [batch(200, warmup=True), batch(110)]
        row = compare.compare(baseline, candidate)["speed"][0]
        self.assertAlmostEqual(row["improvement_fraction"], 0.1)
        self.assertFalse(row["target_met_observed"])
        self.assertFalse(row["enough_repetitions"])

    def test_plain_and_speculative_rates_are_separate_but_exactness_is_shared(self):
        out = compare.compare([batch(40, False), batch(80)], [batch(44, False), batch(104)])
        rates = {tuple(row["draft_modes"]): row for row in out["speed"]}
        self.assertEqual(rates[(False,)]["baseline_samples"], [40])
        self.assertEqual(rates[(True,)]["candidate_samples"], [104])
        self.assertAlmostEqual(rates[(False,)]["improvement_fraction"], 0.1)
        self.assertAlmostEqual(rates[(True,)]["improvement_fraction"], 0.3)
        self.assertTrue(all(row["equal"] for row in out["accuracy"]))
        self.assertTrue(compare.token_equal(result([1], draft=False), result([1], draft=True)))

    def test_bad_rates_and_unattributed_batches_never_enter_speed_medians(self):
        invalid = [batch(value) for value in (None, float("nan"), float("inf"), -1, 0, True, "100")]
        invalid += [batch(1000, metrics={"valid": False}), batch(1000, metrics=None),
                    batch(1000, runs=[result([1], complete=False)]),
                    batch(1000, runs=[result([1], draft=None)])]
        out = compare.compare([batch(100)], invalid + [batch(110)])
        self.assertEqual(out["speed"][0]["candidate_samples"], [110])
        self.assertEqual(out["speed_exclusions"]["candidate"],
                         {"invalid_rate": 7, "invalid_or_missing_counter_attribution": 2,
                          "incomplete_or_failed_request": 1, "unknown_draft_mode": 1})


class Calibration(unittest.TestCase):
    def test_warmup_and_outlier_do_not_dominate_median(self):
        records = [event(1, i, 10**9 if i < 5 else 10**7) for i in range(12)]
        records += [event(1, 12, 5 * 10**9)]
        profile = fit.fit(records, min_samples=3, max_drafts=0)
        self.assertEqual(profile["profiles"][0]["forward_ms"], [10])
        self.assertEqual(profile["profiles"][0]["overhead_ms"], 10)
        self.assertEqual(profile["provenance"]["excluded"], {"request_warmup": 3, "shape_warmup": 2})

    def test_missing_width_is_not_interpolated(self):
        records = [event(row, i) for row in (1, 2, 4, 5, 6) for i in range(12)]
        profile = fit.fit(records)
        self.assertEqual(profile["profiles"], [])
        self.assertEqual(profile["provenance"]["coverage"][0]["missing_rows"], [3])

    def test_proposed_end_selects_context_range(self):
        records = [event(1, i, context=4096) for i in range(12)]
        profile = fit.fit(records, ranges=((2048, 4096), (4097, 8192)), max_drafts=0)
        self.assertEqual(profile["profiles"][0]["min_context"], 4097)

    def test_proposed_end_must_not_straddle_lower_profile_boundary(self):
        records = [event(4, i, context=4094) for i in range(12)]
        out = fit.fit(records, ranges=((2048, 4096), (4097, 8192)), max_drafts=3)
        self.assertEqual(out["profiles"], [])
        self.assertEqual(out["provenance"]["excluded"]["straddles_profile_range"], 9)

    def test_plain_and_bucket_transition_are_excluded(self):
        plain = event(1, 10)
        plain["windows"][0]["drafts_enabled"] = False
        transition = event(1, 10)
        transition["windows"][0]["bucket_changed"] = True
        out = fit.fit([plain, transition], skip_shape=0, min_samples=1, max_drafts=0)
        self.assertEqual(out["profiles"], [])
        self.assertEqual(out["provenance"]["excluded"], {"not_draft_enabled": 1, "bucket_transition": 1})

    def test_overlapping_inclusive_ranges_rejected(self):
        with self.assertRaises(ValueError):
            fit.fit([], ranges=((1, 2048), (2048, 4096)))

    def test_overhead_does_not_include_release_prefill_gap(self):
        first = [event(1, i) for i in range(12)]
        second = [event(1, i) for i in range(12)]
        for row in second:
            row["start_ns"] += 10**12
            row["windows"][0]["trace_id"] = 2
        out = fit.fit(first + [{"event": "release"}] + second, max_drafts=0)
        self.assertEqual(out["profiles"][0]["overhead_ms"], 10)

    def test_missing_round_clock_never_reuses_biased_legacy_overhead(self):
        records = [event(1, i) for i in range(12)]
        for row in records:
            row.pop("start_ns")
        out = fit.fit(records, max_drafts=0)
        self.assertEqual(out["profiles"], [])
        self.assertEqual(out["provenance"]["coverage"][0]["overhead_samples"], 0)


class FixtureConstruction(unittest.TestCase):
    def test_exact_length_with_three_records(self):
        args = ([1], [2], [3], [4], [[5, 6, 7], [8, 9]], [[20], [21, 22], [23]], 2048)
        ids, positions = fixtures.assemble(*args)
        self.assertEqual(len(ids), 2048)
        self.assertEqual([ids[position] for position in positions], [20, 21, 23])
        self.assertEqual(fixtures.assemble(*args), (ids, positions))

    def test_near_one_million_has_output_and_verifier_reserve(self):
        self.assertLessEqual(1047040 + 1024 + 18, 1048576)
        ids, _ = fixtures.assemble([1], [2], [3], [4], [[5] * 100], [], 1047040)
        self.assertEqual(len(ids), 1047040)


if __name__ == "__main__":
    unittest.main()
