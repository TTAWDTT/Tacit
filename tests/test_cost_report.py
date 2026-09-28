"""Offline contract checks for the tlu.costs aggregator."""

from __future__ import annotations

import copy
import unittest

from tools.cost_report import RecordError, aggregate
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
    if version == "tlu.costs.v2":
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

        comparison = paired_report([left, right], replicates=100)["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 1)
        self.assertFalse(comparison["control_alignment"]["model_strata_matched"])
        self.assertNotIn("model_population_id", comparison["task_stratum"])


if __name__ == "__main__":
    unittest.main()
