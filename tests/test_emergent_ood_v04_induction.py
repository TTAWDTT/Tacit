from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.emergent_ood_v0_4.induce_protocol_cards import (
    OUTPUT_SCHEMA,
    build_induction_messages,
    parse_candidate_cards,
    run_induction,
    sample_training_examples,
)
from experiments.emergent_ood_v0_4.split import build_split


class EmergentOODProtocolInductionTests(unittest.TestCase):
    def setUp(self):
        self.split = build_split(seed=31)
        self.key = bytes(range(32))

    def test_induction_examples_are_reproducible_and_training_only(self):
        examples = sample_training_examples(split=self.split, task_key=self.key, example_count=32)
        self.assertEqual(examples, sample_training_examples(split=self.split, task_key=self.key, example_count=32))
        train_ids = set(self.split["train_meaning_ids"])
        heldout_ids = set(self.split["held_out_meaning_ids"])
        example_ids = {
            "m-" + hashlib.sha256(json.dumps(
                [example[name] for name in self.split["attributes"]],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:16]
            for example in examples
        }
        self.assertEqual(len(example_ids), 32)
        self.assertTrue(example_ids <= train_ids)
        self.assertFalse(example_ids & heldout_ids)
        messages = build_induction_messages(split=self.split, examples=examples, candidate_count=3)
        payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
        self.assertEqual(payload["sampled_training_meanings_only"], examples)
        self.assertIn("held-out compositions", payload["task"])
        self.assertNotIn("split_seed", payload)

    def test_parser_hashes_unique_cards_and_rejects_duplicate_or_extra_fields(self):
        response = json.dumps({
            "schema": OUTPUT_SCHEMA,
            "protocols": [
                {"sender_instruction": "Emit an unambiguous reusable code.", "receiver_instruction": "Decode each field and match the tuple."},
                {"sender_instruction": "Use another compositional code.", "receiver_instruction": "Decode the alternate code and match."},
            ],
        })
        cards = parse_candidate_cards(response, candidate_count=2)
        self.assertEqual(len({card["protocol_id"] for card in cards}), 2)
        self.assertTrue(all(card["schema"] == "tlu.shared_protocol_card.v1" for card in cards))
        self.assertEqual(cards, parse_candidate_cards(response, candidate_count=2))
        duplicate = json.dumps({
            "schema": OUTPUT_SCHEMA,
            "protocols": [
                {"sender_instruction": "same", "receiver_instruction": "same"},
                {"sender_instruction": "same", "receiver_instruction": "same"},
            ],
        })
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_candidate_cards(duplicate, candidate_count=2)
        with self.assertRaisesRegex(ValueError, "unexpected schema"):
            parse_candidate_cards(json.dumps({"schema": OUTPUT_SCHEMA, "protocols": [], "rationale": "bad"}), candidate_count=0)

    def test_dry_run_reports_training_only_plan_without_writing_or_calling(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            root = Path(temporary)
            key_path = root / "task.key"
            key_path.write_bytes(self.key)
            output_dir = root / "induced"
            plan = run_induction(
                split_seed=31, task_key_path=key_path, example_count=16, candidate_count=2,
                output_dir=output_dir, execute=False,
            )
            self.assertEqual(plan["mode"], "dry_run")
            self.assertFalse(plan["inference_started"])
            self.assertFalse(plan["model_loaded"])
            self.assertEqual(plan["test_examples_in_prompt"], 0)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
