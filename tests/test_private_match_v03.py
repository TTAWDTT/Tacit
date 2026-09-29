from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tacit import ChatCompletion, exchange_dialogue


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "private_match_v0_3"
    / "generate_tasks.py"
)
SPEC = importlib.util.spec_from_file_location("private_match_v0_3", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class CoordinateOracle:
    def __init__(self, coordinate: str) -> None:
        self.coordinate = coordinate
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[1]["content"])
        view = json.loads(request["private_context"])
        self_value = view["private_value"]
        self.assert_role(view)
        return ChatCompletion(f"{self.coordinate}={self_value}", f"oracle-{self.coordinate}")

    def assert_role(self, view):
        if view["coordinate"] != self.coordinate:
            raise AssertionError("sender received the other role's private view")


class CandidateTableOracle:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[1]["content"])
        receiver = json.loads(request["private_context"])
        values = {}
        for entry in request["visible_transcript"]:
            coordinate, value = entry["message"].split("=", 1)
            values[coordinate] = value
        answer = module.oracle_answer(receiver, values["x"], values["y"])
        return ChatCompletion(answer, "oracle-receiver")


class PrivateMatchV03Tests(unittest.TestCase):
    def test_generation_is_deterministic_and_role_separated(self):
        first = module.generate_episode(episode_id="triad-1", seed=42, q=4)
        second = module.generate_episode(episode_id="triad-1", seed=42, q=4)
        self.assertEqual(first, second)
        sender_x, sender_y, receiver, gold = first
        self.assertTrue(sender_x["private_value"].startswith("x"))
        self.assertNotIn("private_value", receiver)
        self.assertNotIn("target_candidate_id", sender_x)
        self.assertNotIn("target_candidate_id", sender_y)
        self.assertNotIn("target_candidate_id", receiver)
        self.assertIn("target_candidate_id", gold)
        self.assertEqual(sender_x["episode_id"], sender_y["episode_id"])

    def test_each_source_alone_leaves_q_candidates_and_both_are_unique(self):
        for q in (2, 4, 8, 16):
            for seed in range(12):
                sender_x, sender_y, receiver, gold = module.generate_episode(
                    episode_id=f"q{q}-seed{seed}", seed=seed, q=q
                )
                x_matches = [
                    row["candidate_id"] for row in receiver["candidates"]
                    if row["record"]["x"] == sender_x["private_value"]
                ]
                y_matches = [
                    row["candidate_id"] for row in receiver["candidates"]
                    if row["record"]["y"] == sender_y["private_value"]
                ]
                self.assertEqual(len(x_matches), q)
                self.assertEqual(len(y_matches), q)
                self.assertEqual(
                    module.oracle_answer(receiver, sender_x["private_value"], sender_y["private_value"]),
                    gold["target_candidate_id"],
                )
                self.assertTrue(module.score_answer(receiver, gold, gold["target_candidate_id"]))
                self.assertFalse(module.score_answer(receiver, gold, "not-a-candidate"))

    def test_exact_bayes_references_and_zero_error_bit_bound(self):
        self.assertEqual(
            module.bayes_accuracy_references(4),
            {
                "no_message": Fraction(1, 16),
                "sender_x_only": Fraction(1, 4),
                "sender_y_only": Fraction(1, 4),
                "both_sources": Fraction(1, 1),
            },
        )
        for q in (2, 4, 8, 16):
            self.assertEqual(2 * (q.bit_length() - 1), (q * q).bit_length() - 1)
        for q in (0, 1, 3, 6, True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                module.bayes_accuracy_references(q)

    def test_validator_rejects_cross_role_leaks_and_incomplete_candidate_table(self):
        sender_x, sender_y, receiver, gold = module.generate_episode(
            episode_id="validate-1", seed=7, q=4
        )
        leaked_x = dict(sender_x, target_candidate_id=gold["target_candidate_id"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            module.validate_episode(leaked_x, sender_y, receiver, gold)
        missing_row_receiver = dict(receiver, candidates=receiver["candidates"][:-1])
        with self.assertRaisesRegex(ValueError, "exactly q squared"):
            module.validate_episode(sender_x, sender_y, missing_row_receiver, gold)

    def test_three_agent_oracle_uses_real_unicast_channel_and_sealed_submission(self):
        sender_x, sender_y, receiver, gold = module.generate_episode(
            episode_id="loopback-oracle-1", seed=777, q=4
        )
        agents = {
            "sender_x": CoordinateOracle("x"),
            "sender_y": CoordinateOracle("y"),
            "receiver": CandidateTableOracle(),
        }

        class Protocol:
            protocol_id = "triadic-oracle-integration-v1"
            agent_instructions = {
                "sender_x": "Send only your x coordinate.",
                "sender_y": "Send only your y coordinate.",
                "receiver": "Use only your table and routed messages.",
            }

        result = exchange_dialogue(
            agents,
            protocol=Protocol(),
            private_contexts={
                "sender_x": json.dumps(sender_x),
                "sender_y": json.dumps(sender_y),
                "receiver": json.dumps(receiver),
            },
            schedule=(("sender_x", "receiver"), ("sender_y", "receiver")),
            task="Find the candidate row matching both private coordinates.",
            max_turns=2,
            final_answer_agent="receiver",
            final_answer_instruction="Return only the exact candidate ID for scoring.",
        )

        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.stop_reason, "schedule_complete")
        self.assertEqual(len(result.transmission_records()), 2)
        self.assertEqual(
            [row["recipients"] for row in result.transmission_records()],
            [["receiver"], ["receiver"]],
        )
        self.assertTrue(module.score_answer(receiver, gold, result.final_submission.text))
        self.assertEqual(
            [row["stage"] for row in result.model_call_records()],
            ["dialogue_turn", "dialogue_turn", "final_answer"],
        )
        self.assertNotIn(result.final_submission.text, [
            turn.completion.text for turn in result.turns
        ])

    def test_dataset_manifest_hashes_roles_and_protects_existing_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shard"
            manifest = module.generate_dataset(output, episodes=5, seed=3000, q=4)
            self.assertEqual(manifest["agent_count"], 3)
            self.assertEqual(manifest["candidate_count"], 16)
            for name in module.ROLE_FILES:
                path = output / name
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    manifest["files_sha256"][name],
                )
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 5)
            with self.assertRaises(FileExistsError):
                module.generate_dataset(output, episodes=5, seed=3000, q=4)
            replaced = module.generate_dataset(output, episodes=2, seed=4000, q=2, force=True)
            self.assertEqual(replaced["episodes"], 2)
            with (output / "receiver.jsonl").open(encoding="utf-8") as stream:
                receiver = json.loads(next(stream))
            self.assertEqual(len(receiver["candidates"]), 4)


if __name__ == "__main__":
    unittest.main()
