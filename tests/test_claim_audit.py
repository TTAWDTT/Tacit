"""Contract tests for exact task-grounded atomic-claim summaries."""

from __future__ import annotations

import copy
import unittest

from tools.claim_audit import ClaimAuditError, aggregate


def _record(episode_id: str, protocol_id: str, claims: list[dict] | None) -> dict:
    messages = [] if claims is None else [{
        "message_id": "m1",
        "sender_id": "agent-a",
        "recipient_id": "agent-b",
        "claims": claims,
    }]
    return {
        "schema_version": "tlu.claim-audit.v1",
        "episode_id": episode_id,
        "protocol_id": protocol_id,
        "claim_extractor_id": "oracle-fact-labels-v1",
        "stratum": {
            "experiment_id": "claim-test",
            "task_id": "private-facts-v1",
            "split": "test",
            "task_parameters": {"fact_count": 3},
            "model_population_id": "synthetic",
            "sender_model": "sender-v1",
            "receiver_model": "oracle-v1",
            "scorer_id": "exact-v1",
        },
        "messages": messages,
    }


def _claim(
    instance_id: str,
    *,
    observed: bool = True,
    true: bool = True,
    known: bool = False,
    relevant: bool = True,
) -> dict:
    return {
        "claim_instance_id": instance_id,
        "fact_id": f"fact-{instance_id}",
        "sender_observed": observed,
        "fact_true": true,
        "receiver_knows_before": known,
        "task_relevant": relevant,
    }


class ClaimAuditTests(unittest.TestCase):
    def test_grounded_novel_relevant_rate_requires_all_four_conditions(self) -> None:
        record = _record("ep1", "format-a", [
            _claim("good"),
            _claim("unobserved", observed=False),
            _claim("false", true=False),
            _claim("known", known=True),
            _claim("irrelevant", relevant=False),
        ])
        group = aggregate([record])["groups"][0]
        self.assertEqual(group["claims"], 5)
        self.assertEqual(group["claim_rates"]["grounded_novel_relevant"], {
            "rate": 0.2, "numerator": 1, "denominator": 5,
        })
        self.assertEqual(group["claim_rates"]["supported"]["numerator"], 3)
        self.assertEqual(group["claim_rates"]["receiver_novel"]["numerator"], 4)

    def test_pooled_and_episode_macro_rates_are_both_reported(self) -> None:
        first = _record("ep1", "format-a", [_claim("a"), _claim("b", true=False)])
        second = _record("ep2", "format-a", [_claim("c")])
        group = aggregate([first, second])["groups"][0]
        self.assertEqual(group["claim_rates"]["grounded_novel_relevant"]["rate"], 2 / 3)
        self.assertEqual(group["macro_episode_grounded_novel_relevant_rate"], 0.75)
        self.assertEqual(group["episodes_with_claims_for_macro"], 2)

    def test_silence_is_preserved_and_undefined_rates_are_null(self) -> None:
        group = aggregate([_record("silent", "no-message", None)])["groups"][0]
        self.assertEqual(group["episodes_without_messages"], 1)
        self.assertEqual(group["episodes_without_claims"], 1)
        self.assertIsNone(group["claim_rates"]["grounded_novel_relevant"]["rate"])
        self.assertIsNone(group["macro_episode_grounded_novel_relevant_rate"])

    def test_strata_and_extractors_are_never_pooled(self) -> None:
        one = _record("same-id", "format-a", [_claim("a")])
        two = copy.deepcopy(one)
        two["stratum"]["receiver_model"] = "other-model"
        three = copy.deepcopy(one)
        three["claim_extractor_id"] = "human-adjudication-v1"
        four = copy.deepcopy(one)
        four["stratum"]["task_parameters"]["fact_count"] = 4
        self.assertEqual(aggregate([one, two, three, four])["group_count"], 4)

    def test_non_finite_task_parameters_are_rejected(self) -> None:
        record = _record("ep1", "format-a", [])
        record["stratum"]["task_parameters"]["bad"] = float("nan")
        with self.assertRaisesRegex(ClaimAuditError, "finite JSON values"):
            aggregate([record])

    def test_duplicate_episode_within_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimAuditError, "duplicate episode_id"):
            aggregate([_record("ep1", "format-a", []), _record("ep1", "format-a", [])])

    def test_invalid_boolean_and_duplicate_claim_instance_are_rejected(self) -> None:
        invalid = _record("ep1", "format-a", [_claim("a")])
        invalid["messages"][0]["claims"][0]["fact_true"] = 1
        with self.assertRaisesRegex(ClaimAuditError, "fact_true: expected boolean"):
            aggregate([invalid])

        duplicate = _record("ep2", "format-a", [_claim("a"), _claim("a")])
        with self.assertRaisesRegex(ClaimAuditError, "duplicate 'a'"):
            aggregate([duplicate])

    def test_report_warns_against_utility_or_superiority_interpretation(self) -> None:
        report = aggregate([_record("ep1", "format-a", [_claim("a")])])
        self.assertIn("not task utility", report["interpretation"])
        self.assertIn("language-superiority", report["interpretation"])


if __name__ == "__main__":
    unittest.main()
