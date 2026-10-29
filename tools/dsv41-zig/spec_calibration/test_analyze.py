"""Guard trace analysis against fabricated coverage, confidence tails, cross-request joins and speed claims."""

import copy
import math
import unittest

import analyze


def catalog(costs=None, streams=1, low=2048, high=4096):
    return {"schema_version": 1, "family": "dsv41", "profiles": [
        {"min_context": low, "max_context": high, "streams": streams,
         "forward_ms": costs or [40, 80, 120, 160, 200, 240], "overhead_ms": 8}]}


def events(selected=2, accepted=1, full=5, context=2048, logits=None, source="a", trace_id=1):
    logits = [math.log(4)] * full if logits is None else logits
    window = {"trace_id": trace_id, "slot": 0, "context": context, "rows": full,
              "held_drafts": full, "selected_drafts": full, "drafts_enabled": True,
              "round_index": 10, "confidence_logits": logits}
    draft = {"event": "draft", "_source": source, "streams": 1, "rows": full,
             "pass_ns": 4_000_000, "total_ns": 5_000_000, "windows": [window]}
    verify = {"event": "verify", "_source": source, "streams": 1, "rows": selected+1, "bucket": 4096,
              "greedy": True, "verify_ns": 100_000_000, "send_ns": 1_000_000, "readback_ns": 2_000_000,
              "sample_ns": 3_000_000, "total_ns": 106_000_000,
              "windows": [{**window, "rows": selected+1, "held_drafts": selected, "host_drafts": 0,
                           "selected_drafts": selected, "potential_accepted_drafts": accepted,
                           "confidence_logits": logits[:selected]}]}
    return draft, verify


def report(records, profile=None):
    return analyze.analyze(records, profile or catalog(), skip_rounds=0, skip_shape=0)


class Choices(unittest.TestCase):
    def test_greedy_search_matches_known_single_and_shared_decisions(self):
        probability = [[0.9, 0.8]]
        self.assertEqual(analyze.choose(probability, [10, 11, 12], 0)["counts"], [2])
        self.assertEqual(analyze.choose(probability, [10, 20, 30], 0)["counts"], [0])
        self.assertEqual(analyze.choose([[0.9], [0.9]], [10, 10, 10, 100], 0)["counts"], [1, 0])

    def test_missing_costs_and_invalid_probabilities_are_not_invented(self):
        for costs in ([10], [10, 0], [10, float("nan")]):
            with self.assertRaises(ValueError):
                analyze.choose([[0.8]], costs, 0)
        with self.assertRaises(ValueError):
            analyze.choose([[0.5, 0.8]], [10, 11, 12], 0)

    def test_sigmoid_extremes_are_stable_and_unknowns_rejected(self):
        self.assertEqual(analyze.survivals([1000, -1000]), [1, 0])
        for logits in ([None], [float("nan")], [True]):
            self.assertIsNone(analyze.survivals(logits))


class Measurements(unittest.TestCase):
    def test_components_acceptance_and_observed_prefix_calibration(self):
        first = events(full=2, logits=[0, 0])
        second = events(full=2, logits=[0, 0], trace_id=2)
        second[1]["verify_ns"] = 200_000_000
        group = report([*first, *second])["groups"][0]
        self.assertEqual(group["component_ms"]["verify"]["median"], 150)
        self.assertEqual(group["component_ms"]["preceding_draft_total"]["median"], 5)
        self.assertEqual(group["selected_k_histogram"], {2: 2})
        self.assertEqual(group["potential_acceptance_by_verified_k"][0]["mean_potential_accepted_drafts"], 1)
        calibration = group["prefix_calibration"]
        self.assertEqual([cell["depth"] for cell in calibration], [1, 2])
        self.assertEqual(calibration[0]["predicted_prefix_survival"], 0.5)
        self.assertEqual(calibration[0]["observed_prefix_survival"], 1)
        self.assertEqual(calibration[1]["observed_prefix_survival"], 0)
        self.assertEqual(calibration[1]["brier_score"], 0.0625)

    def test_no_unverified_acceptance_tail_or_missing_component_fill(self):
        draft, verify = events(selected=1, accepted=1)
        verify.pop("readback_ns")
        group = report([draft, verify])["groups"][0]
        self.assertEqual(len(group["prefix_calibration"]), 1)
        self.assertNotIn("readback", group["component_ms"])
        self.assertEqual(group["excluded"]["invalid_or_missing_readback_ns"], 1)

    def test_invalid_confidences_do_not_discard_real_component_measurements(self):
        draft, verify = events()
        draft["windows"][0]["confidence_logits"][0] = None
        group = report([draft, verify])["groups"][0]
        self.assertEqual(group["component_ms"]["preceding_draft_pass"]["median"], 4)
        self.assertEqual(group["policy"]["eligible_rounds"], 0)

    def test_warmups_and_bucket_transition_are_explicit(self):
        draft, verify = events()
        verify["windows"][0]["first_round"] = True
        result = report([draft, verify])
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["excluded"]["request_warmup"], 1)
        verify["windows"][0]["first_round"] = False
        verify["windows"][0]["bucket_changed"] = True
        self.assertEqual(report([verify])["excluded"]["bucket_transition"], 1)


class Replay(unittest.TestCase):
    def test_full_held_vector_changes_choice_without_claiming_actual_tps(self):
        group = report(list(events(selected=1, accepted=1)))["groups"][0]
        policy = group["policy"]
        self.assertEqual(policy["legacy_selected_rows"], {6: 1})
        self.assertEqual(policy["measured_selected_rows"], {1: 1})
        self.assertTrue(policy["changed_fraction"] == 1)
        self.assertGreater(policy["modeled_gain_on_replayed_states"], 0.2)
        self.assertEqual(policy["label"], "predicted_counterfactual_not_actual_tps")
        self.assertNotIn("actual_tps", policy)

    def test_truncated_verify_vector_is_never_used_as_full_proposal(self):
        _, verify = events(selected=1, accepted=1)
        group = report([verify])["groups"][0]
        self.assertEqual(group["policy"]["eligible_rounds"], 0)
        self.assertEqual(group["excluded"]["missing_or_stale_preceding_draft"], 1)

    def test_source_context_and_release_prevent_stale_joins(self):
        for change in ("source", "context", "release", "header"):
            draft, verify = events()
            records = [draft, verify]
            if change == "source":
                verify["_source"] = "other"
            elif change == "context":
                verify["windows"][0]["context"] += 1
            else:
                records.insert(1, {"event": change, "_source": "a", "windows": verify["windows"]})
            self.assertEqual(report(records)["groups"][0]["policy"]["eligible_rounds"], 0, change)

    def test_full_proposal_boundary_and_missing_rows_prevent_extrapolation(self):
        records = list(events(selected=1, accepted=1, context=4092))
        group = report(records)["groups"][0]
        self.assertEqual(group["excluded"]["no_profile_for_full_proposal_span"], 1)
        group = report(list(events()), catalog(costs=[40]*5))["groups"][0]
        self.assertEqual(group["excluded"]["incomplete_feasible_row_costs"], 1)
        profile = catalog()
        profile["profiles"][0].pop("overhead_ms")
        self.assertEqual(report(list(events()), profile)["groups"][0]["excluded"]["missing_measured_overhead"], 1)

    def test_shared_draft_timing_is_counted_once(self):
        draft, verify = events(full=1, selected=1, accepted=1)
        for record in (draft, verify):
            record["streams"] = 2
            record["rows"] *= 2
            second = copy.deepcopy(record["windows"][0])
            second.update(trace_id=2, slot=1)
            record["windows"].append(second)
        group = report([draft, verify], catalog(costs=[40, 80, 120, 160], streams=2))["groups"][0]
        self.assertEqual(group["component_ms"]["preceding_draft_total"]["median"], 5)
        self.assertEqual(group["policy"]["eligible_rounds"], 1)

    def test_sweep_choices_do_not_claim_legacy_runtime_agreement(self):
        draft, verify = events()
        verify["calibration_k"] = 2
        group = report([draft, verify])["groups"][0]
        self.assertEqual(group["sweep_events"], 1)
        self.assertIsNone(group["policy"]["legacy_scenario_matches_observed_fraction"])

    def test_invalid_catalog_and_incomplete_sources_are_visible(self):
        profile = catalog()
        profile["profiles"].append(copy.deepcopy(profile["profiles"][0]))
        with self.assertRaises(ValueError):
            report([], profile)
        records = list(events())
        self.assertFalse(report(records)["trace_complete_by_source"]["a"])
        records.append({"event": "summary", "_source": "a", "complete": True})
        self.assertTrue(report(records)["trace_complete_by_source"]["a"])
        records.append(events()[1])
        self.assertFalse(report(records)["trace_complete_by_source"]["a"])


if __name__ == "__main__":
    unittest.main()
