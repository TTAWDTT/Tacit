"""Offline contract checks for the tlu.costs aggregator."""

from __future__ import annotations

import copy
import unittest

from tools.cost_report import RecordError, aggregate
from tools.frontier_report import frontier_report
from tools.paired_report import paired_report


def _record(episode_id: str, *, version: str = "tlu.costs.v2", framing: object = 0) -> dict:
    transmission = {
        "round": 1,
        "sender": "sender",
        "recipients": ["receiver"],
        "payload_utf8_bytes": 3,
        "recipient_tokens": {
            "receiver": {"tokenizer": "test-tokenizer-v1", "tokens": 2}
        },
    }
    if version == "tlu.costs.v3":
        transmission.pop("payload_utf8_bytes")
        transmission.update({
            "payload_bytes": 12,
            "framing_bytes": 4,
            "encoding": "float16-le",
            "media_type": "application/vnd.tlu.tensor",
            "transport_boundary": "network",
            "payload_metadata": {"dtype": "float16", "shape": [2, 3]},
        })
    if framing is not _MISSING:
        transmission["framing_utf8_bytes"] = framing

    record = {
        "schema_version": version,
        "episode_id": episode_id,
        "protocol": {"policy_id": "fixed", "code_id": "plain-v1", "decoder_id": "direct-v1"},
        "outcome": {"joint_success": True, "answer_score": 1.0},
        "transmissions": [transmission],
        "model_calls": [],
        "runtime": {"wall_seconds": 5.0, "critical_path_seconds": 2.0},
        "setup": [],
    }
    if version in {"tlu.costs.v2", "tlu.costs.v3"}:
        record["stratum"] = {
            "experiment_id": "unit-test",
            "task_id": "toy@1",
            "split": "test",
            "task_parameters": {"n": 1},
            "model_population_id": "deterministic-test",
            "agent_models": {"sender": "oracle-v1", "receiver": "oracle-v1"},
            "scorer_id": "exact-v1",
        }
    return record


_MISSING = object()


class CostReportContractTests(unittest.TestCase):
    def test_v3_records_binary_payloads_and_metadata(self) -> None:
        record = _record("binary-episode", version="tlu.costs.v3")
        group = aggregate([record])["groups"][0]
        self.assertEqual(group["channel"]["payload_bytes"]["observed_sum"], 12)
        self.assertEqual(group["channel"]["wire_bytes"]["observed_sum"], 16)

    def test_v3_requires_encoding_and_actual_framing_bytes(self) -> None:
        missing = _record("binary-episode", version="tlu.costs.v3")
        del missing["transmissions"][0]["encoding"]
        with self.assertRaisesRegex(RecordError, "encoding: expected non-empty string"):
            aggregate([missing])
        missing = _record("binary-episode", version="tlu.costs.v3")
        del missing["transmissions"][0]["framing_bytes"]
        with self.assertRaisesRegex(RecordError, "framing_bytes: value is required"):
            aggregate([missing])

    def test_v3_rejects_unserialized_same_process_boundary(self) -> None:
        record = _record("binary-episode", version="tlu.costs.v3")
        record["transmissions"][0]["transport_boundary"] = "same_process"
        with self.assertRaisesRegex(RecordError, "transport_boundary: expected"):
            aggregate([record])

    def test_v3_is_supported_by_paired_and_frontier_reports(self) -> None:
        left = _record("binary-episode", version="tlu.costs.v3")
        left["protocol"]["code_id"] = "binary"
        right = copy.deepcopy(left)
        right["protocol"]["code_id"] = "smaller-binary"
        right["transmissions"][0]["payload_bytes"] = 8
        paired = paired_report([left, right], replicates=100, seed=3)
        self.assertEqual(paired["comparisons"][0]["metrics"]["wire_bytes"]["mean_left_minus_right"], 4.0)
        frontier = frontier_report([left, right])
        self.assertEqual(frontier["input_schema_version"], "tlu.costs.v3")

    def test_v2_requires_framing_byte_count(self) -> None:
        with self.assertRaisesRegex(RecordError, "framing_utf8_bytes: value is required"):
            aggregate([_record("episode-1", framing=_MISSING)])

    def test_v2_rejects_unknown_framing_byte_count(self) -> None:
        with self.assertRaisesRegex(RecordError, "framing_utf8_bytes: value is required"):
            aggregate([_record("episode-1", framing=None)])

    def test_v2_accepts_zero_framing_and_reports_wire_cost(self) -> None:
        group = aggregate([_record("episode-1")])["groups"][0]
        self.assertEqual(group["channel"]["wire_bytes"]["observed_sum"], 3)

    def test_critical_path_is_aggregated_as_runtime(self) -> None:
        second = copy.deepcopy(_record("episode-2"))
        second["runtime"]["critical_path_seconds"] = 4.0
        group = aggregate([_record("episode-1"), second])["groups"][0]
        self.assertEqual(group["runtime"]["critical_path_seconds"]["observed_sum"], 6.0)
        self.assertEqual(group["runtime"]["critical_path_seconds"]["observed_mean"], 3.0)

    def test_legacy_v1_still_accepts_missing_framing(self) -> None:
        group = aggregate([_record("legacy-episode", version="tlu.costs.v1", framing=_MISSING)])[
            "groups"
        ][0]
        self.assertEqual(group["aggregation_scope"], "protocol_only_legacy_v1")
        self.assertFalse(group["channel"]["wire_bytes"]["complete"])

    def test_paired_report_matches_episodes_and_uses_left_minus_right(self) -> None:
        left_one = _record("episode-1")
        left_one["protocol"]["code_id"] = "left"
        left_two = _record("episode-2")
        left_two["protocol"]["code_id"] = "left"
        left_two["outcome"]["joint_success"] = False

        right_one = _record("episode-1")
        right_one["protocol"]["code_id"] = "right"
        right_one["outcome"]["joint_success"] = False
        right_two = _record("episode-2")
        right_two["protocol"]["code_id"] = "right"
        right_two["outcome"]["joint_success"] = False
        right_two["runtime"]["critical_path_seconds"] = 4.0

        report = paired_report([left_one, left_two, right_one, right_two], replicates=200, seed=7)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 2)
        self.assertEqual(comparison["metrics"]["joint_success"]["mean_left_minus_right"], 0.5)
        self.assertEqual(comparison["metrics"]["critical_path_seconds"]["mean_left_minus_right"], -1.0)
        self.assertEqual(comparison["control_alignment"], {
            "model_strata_matched": True,
            "policy_matched": True,
            "decoder_matched": True,
            "code_differs": True,
        })

    def test_paired_report_is_reproducible_and_reports_unmatched_episodes(self) -> None:
        left = _record("shared")
        left["protocol"]["code_id"] = "left"
        left_only = _record("left-only")
        left_only["protocol"]["code_id"] = "left"
        right = _record("shared")
        right["protocol"]["code_id"] = "right"
        right_only = _record("right-only")
        right_only["protocol"]["code_id"] = "right"

        records = [left, left_only, right, right_only]
        first = paired_report(records, replicates=200, seed=11)
        second = paired_report(records, replicates=200, seed=11)
        self.assertEqual(first, second)
        comparison = first["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 1)
        self.assertEqual(comparison["left_only_episodes"], 1)
        self.assertEqual(comparison["right_only_episodes"], 1)

    def test_paired_report_matches_task_cells_and_marks_model_mismatch(self) -> None:
        left = _record("same-task")
        left["protocol"]["code_id"] = "natural-language"
        right = _record("same-task")
        right["protocol"]["code_id"] = "compact-code"
        right["stratum"]["model_population_id"] = "different-model-pair"
        right["stratum"]["agent_models"] = {"sender": "model-b", "receiver": "model-b"}
        left["model_calls"] = [{
            "agent": "receiver", "stage": "answer", "model": "model-a",
            "tokenizer": "tokenizer-a", "input_tokens": 12, "output_tokens": 2,
            "service_seconds": 1.0, "retry": False, "truncated": False,
        }]
        right["model_calls"] = [{
            "agent": "receiver", "stage": "answer", "model": "model-b",
            "tokenizer": "tokenizer-b", "input_tokens": 10, "output_tokens": 2,
            "service_seconds": 1.0, "retry": False, "truncated": False,
        }]

        comparison = paired_report([left, right], replicates=100)["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 1)
        self.assertFalse(comparison["control_alignment"]["model_strata_matched"])
        self.assertNotIn("model_population_id", comparison["task_stratum"])
        self.assertFalse(comparison["tokenizer_units"]["deltas_comparable"])
        token_metric = comparison["metrics"]["input_tokens"]
        self.assertEqual(token_metric["paired_episodes"], 0)
        self.assertEqual(token_metric["missing_pairs"], 1)

    def test_frontier_reports_nondominated_points_without_scalarizing_costs(self) -> None:
        low_cost = _record("episode-1")
        low_cost["protocol"]["code_id"] = "low-cost"
        low_cost["transmissions"][0]["payload_utf8_bytes"] = 1
        low_cost["runtime"]["wall_seconds"] = 2.0

        dominated = _record("episode-1")
        dominated["protocol"]["code_id"] = "dominated"
        dominated["transmissions"][0]["payload_utf8_bytes"] = 3
        dominated["runtime"]["wall_seconds"] = 3.0

        report = frontier_report([low_cost, dominated])
        group = report["groups"][0]
        operational = next(item for item in group["frontiers"] if item["scope"] == "operational")
        self.assertEqual(operational["minimize"], [
            "wire_bytes", "input_tokens", "output_tokens", "model_calls",
            "service_seconds", "wall_seconds", "amortized_setup_bytes",
        ])
        self.assertEqual(operational["nondominated_protocols"], [low_cost["protocol"]])

    def test_frontier_refuses_to_compare_conditions_with_different_episode_sets(self) -> None:
        left_one = _record("episode-1")
        left_one["protocol"]["code_id"] = "left"
        left_two = _record("episode-2")
        left_two["protocol"]["code_id"] = "left"
        right = _record("episode-1")
        right["protocol"]["code_id"] = "right"

        group = frontier_report([left_one, left_two, right])["groups"][0]
        self.assertFalse(group["episode_coverage_matched_across_conditions"])
        self.assertTrue(all(frontier["eligible_conditions"] == 0 for frontier in group["frontiers"]))
        self.assertTrue(all(frontier["excluded_conditions"][0]["episode_set_mismatch"] for frontier in group["frontiers"]))

    def test_frontier_does_not_add_token_counts_from_heterogeneous_tokenizers(self) -> None:
        left = _record("episode-1")
        left["protocol"]["code_id"] = "left"
        right = _record("episode-1")
        right["protocol"]["code_id"] = "right"
        for record in (left, right):
            record["stratum"]["model_population_id"] = "heterogeneous-pair"
            record["stratum"]["agent_models"] = {"sender": "model-a", "receiver": "model-b"}
            record["model_calls"] = [
                {"agent": "sender", "stage": "communicate", "model": "model-a",
                 "tokenizer": "tokenizer-a", "input_tokens": 10, "output_tokens": 2,
                 "service_seconds": 1.0, "retry": False, "truncated": False},
                {"agent": "receiver", "stage": "answer", "model": "model-b",
                 "tokenizer": "tokenizer-b", "input_tokens": 12, "output_tokens": 3,
                 "service_seconds": 1.0, "retry": False, "truncated": False},
            ]

        group = frontier_report([left, right])["groups"][0]
        self.assertEqual(group["tokenizer_units"], ["tokenizer-a", "tokenizer-b"])
        self.assertFalse(group["conditions"][0]["costs"]["input_tokens"]["complete"])
        channel_frontier = next(item for item in group["frontiers"] if item["scope"] == "channel_bytes")
        token_frontier = next(item for item in group["frontiers"] if item["scope"] == "channel_and_inference_tokens")
        self.assertEqual(channel_frontier["eligible_conditions"], 2)
        self.assertEqual(token_frontier["eligible_conditions"], 0)

    def test_frontier_refuses_legacy_records_without_task_strata(self) -> None:
        report = frontier_report([_record("legacy", version="tlu.costs.v1", framing=_MISSING)])
        self.assertEqual(report["groups"], [])
        self.assertIn("lacks task and model strata", report["excluded_reason"])

    def test_frontier_deduplicates_setup_cost_over_the_condition_group(self) -> None:
        first = _record("episode-1")
        second = _record("episode-2")
        first["protocol"]["code_id"] = "same-condition"
        second["protocol"]["code_id"] = "same-condition"
        first["setup"] = [{
            "artifact_id": "shared-codebook-v1",
            "one_time_bytes": 100,
            "one_time_tokens": {},
            "reuse_horizon": 10,
        }]

        point = frontier_report([first, second])["groups"][0]["conditions"][0]
        self.assertEqual(point["costs"]["amortized_setup_bytes"]["mean"], 10.0)


if __name__ == "__main__":
    unittest.main()
