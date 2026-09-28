"""Offline contract checks for the tlu.costs aggregator."""

from __future__ import annotations

import copy
import unittest

from tools.cost_report import RecordError, aggregate


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


if __name__ == "__main__":
    unittest.main()
