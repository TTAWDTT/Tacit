from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "private_match_v0_1" / "generate_tasks.py"
SPEC = importlib.util.spec_from_file_location("private_match_v0_1", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class PrivateMatchTaskTests(unittest.TestCase):
    def test_generation_is_deterministic_and_role_separated(self):
        first = module.generate_episode(
            episode_id="task-1", seed=42, candidate_count=8, feature_count=5, vocabulary_size=16
        )
        second = module.generate_episode(
            episode_id="task-1", seed=42, candidate_count=8, feature_count=5, vocabulary_size=16
        )
        self.assertEqual(first, second)
        sender, receiver, gold = first
        self.assertIn("target_record", sender)
        self.assertNotIn("target_record", receiver)
        self.assertNotIn("target_candidate_id", receiver)
        self.assertIn("target_candidate_id", gold)

    def test_each_episode_has_one_unique_exact_match(self):
        for seed in range(20):
            sender, receiver, gold = module.generate_episode(
                episode_id=f"task-{seed}", seed=seed, candidate_count=12,
                feature_count=4, vocabulary_size=5,
            )
            matches = [c["candidate_id"] for c in receiver["candidates"] if c["record"] == sender["target_record"]]
            self.assertEqual(matches, [gold["target_candidate_id"]])
            self.assertEqual(len({tuple(c["record"].values()) for c in receiver["candidates"]}), 12)

    def test_exact_no_message_reference_and_strict_answer_scoring(self):
        self.assertEqual(module.no_message_bayes_accuracy(4), 0.25)
        self.assertEqual(module.no_message_bayes_accuracy(8), 0.125)
        with self.assertRaises(ValueError):
            module.no_message_bayes_accuracy(1)
        sender, receiver, gold = module.generate_episode(
            episode_id="task-2", seed=8, candidate_count=4, feature_count=2, vocabulary_size=4
        )
        self.assertEqual((1, 4), (1, len(receiver["candidates"])))
        self.assertTrue(module.score_answer(receiver, gold, gold["target_candidate_id"]))
        wrong = next(c["candidate_id"] for c in receiver["candidates"] if c["candidate_id"] != gold["target_candidate_id"])
        self.assertFalse(module.score_answer(receiver, gold, wrong))
        self.assertFalse(module.score_answer(receiver, gold, gold["target_candidate_id"] + "."))
        self.assertFalse(module.score_answer(receiver, gold, 1))

    def test_invalid_task_spaces_are_rejected(self):
        with self.assertRaises(ValueError):
            module.generate_episode(
                episode_id="small", seed=0, candidate_count=5, feature_count=1, vocabulary_size=2
            )
        with self.assertRaises(ValueError):
            module.generate_episode(
                episode_id="small", seed=-1, candidate_count=2, feature_count=1, vocabulary_size=2
            )

    def test_dataset_manifest_hashes_and_no_overwrite_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            manifest = module.generate_dataset(
                output, episodes=3, seed=10, candidate_count=4, feature_count=2,
                vocabulary_size=4,
            )
            self.assertEqual(manifest["no_message_bayes_accuracy"], {"numerator": 1, "denominator": 4})
            for filename, digest in manifest["files_sha256"].items():
                self.assertEqual(digest, module._digest(output / filename))
            sender_rows = [json.loads(line) for line in (output / "sender.jsonl").read_text().splitlines()]
            receiver_rows = [json.loads(line) for line in (output / "receiver.jsonl").read_text().splitlines()]
            gold_rows = [json.loads(line) for line in (output / "gold.jsonl").read_text().splitlines()]
            self.assertEqual((len(sender_rows), len(receiver_rows), len(gold_rows)), (3, 3, 3))
            self.assertEqual([row["episode_id"] for row in sender_rows], ["pm-000000", "pm-000001", "pm-000002"])
            self.assertNotIn("seed", sender_rows[0]["episode_id"])
            with self.assertRaises(FileExistsError):
                module.generate_dataset(
                    output, episodes=3, seed=10, candidate_count=4, feature_count=2,
                    vocabulary_size=4,
                )


if __name__ == "__main__":
    unittest.main()
