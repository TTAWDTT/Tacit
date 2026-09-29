from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tacit.runtime import ChatCompletion
from tools.cost_report import aggregate


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "emergent_ood_v0_3" / "runner.py"
SPEC = importlib.util.spec_from_file_location("emergent_ood_runner_v0_3", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class FakeSender:
    model_name = "fake-sender-v1"

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[-1]["content"])
        if set(request) != {"private_target"}:
            raise AssertionError("sender request leaked receiver candidates or gold fields")
        target = request["private_target"]
        text = (
            f"A {target['shape']} object; its color is {target['color']}; "
            f"it appears {module.QUANTITY_PHRASES[{ 'one': 0, 'two': 1, 'three': 2 }[target['quantity']]]}."
        )
        return ChatCompletion(text, self.model_name, 41, 12, 0.01)


class FakeReceiver:
    model_name = "fake-receiver-v1"

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[-1]["content"])
        if set(request) != {"task", "clue", "candidates"}:
            raise AssertionError("receiver request includes fields outside its role view")
        clue = request["clue"]
        candidates = request["candidates"]
        if clue is None:
            choice = candidates[0]["candidate_id"]
        else:
            quantity_phrases = {"one": "once", "two": "twice", "three": "three times"}
            matches = [
                row for row in candidates
                if row["shape"] in clue and row["color"] in clue
                and quantity_phrases[row["quantity"]] in clue
            ]
            choice = matches[0]["candidate_id"] if len(matches) == 1 else candidates[-1]["candidate_id"]
        return ChatCompletion(json.dumps({"candidate_id": choice}), self.model_name, 57, 6, 0.02)


class EmergentOODRunnerTests(unittest.TestCase):
    def setUp(self):
        self.episodes = module.balanced_block(17, 5)

    def _run_all(self, condition, sender=None):
        receiver = FakeReceiver()
        rows = [
            module.run_condition(
                episode=episode,
                condition=condition,
                sender_model=sender,
                receiver_model=receiver,
                sender_tokenizer_id="fake-tokenizer-sender" if sender else None,
                receiver_tokenizer_id="fake-tokenizer-receiver",
                model_population_id="fake-pair-v1",
            )
            for episode in self.episodes
        ]
        self.assertEqual(len(receiver.calls), 5)
        return rows, receiver

    def test_balanced_block_is_one_identical_ordered_set_with_all_targets(self):
        ids = [[row["candidate_id"] for row in episode["receiver"]["candidates"]] for episode in self.episodes]
        self.assertTrue(all(candidate_ids == ids[0] for candidate_ids in ids))
        self.assertEqual(
            {episode["gold"]["target_id"] for episode in self.episodes},
            set(ids[0]),
        )
        self.assertEqual(
            {episode["gold"]["target_position"] for episode in self.episodes},
            set(range(5)),
        )

    def test_full_information_control_is_exact_and_does_not_claim_channel_use(self):
        rows, receiver = self._run_all("full_information")
        self.assertTrue(all(row["outcome"]["joint_success"] for row in rows))
        self.assertTrue(all(row["transmissions"] == [] for row in rows))
        self.assertTrue(all("private_target" not in call[-1]["content"] for call in receiver.calls))
        self.assertTrue(all("clue" in call[-1]["content"] for call in receiver.calls))
        self.assertEqual(sum(len(row["model_calls"]) for row in rows), 5)

    def test_no_message_uniform_choice_scores_exactly_one_of_five(self):
        rows, _ = self._run_all("no_message")
        self.assertEqual(sum(row["outcome"]["joint_success"] for row in rows), 1)
        self.assertTrue(all(row["transmissions"] == [] for row in rows))

    def test_natural_language_delivers_verbatim_message_and_valid_cost_rows(self):
        sender = FakeSender()
        rows, _ = self._run_all("natural_language", sender=sender)
        self.assertTrue(all(row["outcome"]["joint_success"] for row in rows))
        self.assertEqual(len(sender.calls), 5)
        self.assertTrue(all(len(row["transmissions"]) == 1 for row in rows))
        self.assertTrue(all(
            row["transmissions"][0]["payload_metadata"]["logical_text_utf8_bytes"]
            == len(row["diagnostics"]["message_text"].encode("utf-8"))
            for row in rows
        ))
        report = aggregate(rows)
        self.assertEqual(report["groups"][0]["channel"]["wire_bytes"]["observed"], 5)

    def test_conditions_keep_the_hard_batch_call_cap(self):
        self.assertEqual(module.planned_model_calls(5, ["full_information"]), 5)
        self.assertEqual(module.planned_model_calls(5, ["natural_language"]), 10)
        self.assertEqual(module.planned_model_calls(5, ["no_message", "natural_language"]), 15)
        self.assertGreater(
            module.planned_model_calls(5, ["no_message", "natural_language"]),
            module.MAX_MODEL_CALLS_PER_BATCH,
        )

    def test_receiver_choice_parser_is_strict(self):
        self.assertEqual(module._parse_choice('{"candidate_id":"s0-c0-q0"}', ["s0-c0-q0"]), ("s0-c0-q0", True))
        for invalid in (
            "s0-c0-q0",
            '{"candidate_id":"unknown"}',
            '{"candidate_id":"s0-c0-q0","reason":"guess"}',
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(module._parse_choice(invalid, ["s0-c0-q0"]), (None, False))

    def test_capability_ledger_gate_requires_perfect_same_receiver_block(self):
        calibration_receiver = FakeReceiver()
        calibration_episodes = module.balanced_block(18, 5)
        rows = [
            module.run_condition(
                episode=episode,
                condition="full_information",
                sender_model=None,
                receiver_model=calibration_receiver,
                sender_tokenizer_id=None,
                receiver_tokenizer_id="fake-tokenizer-receiver",
                model_population_id="fake-pair-v1",
            )
            for episode in calibration_episodes
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "full_information.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            module.validate_capability_ledger(
                ledger, self.episodes,
                receiver_model="fake-receiver-v1",
                receiver_tokenizer_id="fake-tokenizer-receiver",
                model_population_id="fake-pair-v1",
            )
            overlapping_rows, _ = self._run_all("full_information")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in overlapping_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint from evaluation episodes"):
                module.validate_capability_ledger(
                    ledger, self.episodes,
                    receiver_model="fake-receiver-v1",
                    receiver_tokenizer_id="fake-tokenizer-receiver",
                    model_population_id="fake-pair-v1",
                )
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "later stages require --capability-ledger"):
                module.validate_capability_ledger(
                    None, self.episodes,
                    receiver_model="fake-receiver-v1",
                    receiver_tokenizer_id="fake-tokenizer-receiver",
                    model_population_id="fake-pair-v1",
                )

            failed = [dict(row) for row in rows]
            failed[0] = json.loads(json.dumps(failed[0]))
            failed[0]["outcome"]["joint_success"] = False
            ledger.write_text("".join(json.dumps(row) + "\n" for row in failed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "all five"):
                module.validate_capability_ledger(
                    ledger, self.episodes,
                    receiver_model="fake-receiver-v1",
                    receiver_tokenizer_id="fake-tokenizer-receiver",
                    model_population_id="fake-pair-v1",
                )

            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model differs"):
                module.validate_capability_ledger(
                    ledger, self.episodes,
                    receiver_model="other-receiver-v1",
                    receiver_tokenizer_id="fake-tokenizer-receiver",
                    model_population_id="fake-pair-v1",
                )
            with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                module.validate_capability_ledger(
                    ledger, self.episodes,
                    receiver_model="fake-receiver-v1",
                    receiver_tokenizer_id="other-tokenizer",
                    model_population_id="fake-pair-v1",
                )


if __name__ == "__main__":
    unittest.main()
