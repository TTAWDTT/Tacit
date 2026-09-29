from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "emergent_ood_v0_2" / "episodes.py"
SPEC = importlib.util.spec_from_file_location("emergent_ood_episodes_v0_2", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class EmergentOODEpisodeTests(unittest.TestCase):
    def test_default_five_choice_ledgers_are_deterministic_and_target_balanced(self):
        first = module.generate_ledgers(17)
        second = module.generate_ledgers(17)
        self.assertEqual(first, second)
        self.assertEqual(first["episode_count"], 5 * 126)
        self.assertEqual(first["no_message_bayes_accuracy"], 0.2)
        self.assertEqual(len(first["sender"]), len(first["receiver"]))
        self.assertEqual(len(first["receiver"]), len(first["gold"]))

        targets = [row["target_id"] for row in first["gold"]]
        self.assertEqual(len(set(targets)), 9)
        self.assertEqual({target: targets.count(target) for target in set(targets)}, {target: 70 for target in set(targets)})

    def test_candidate_order_is_constant_within_each_set_and_cannot_encode_target(self):
        ledgers = module.generate_ledgers(19, candidate_count=5)
        orders_by_set = {}
        target_positions_by_set = {}
        for receiver, gold in zip(ledgers["receiver"], ledgers["gold"]):
            candidate_ids = tuple(row["candidate_id"] for row in receiver["candidates"])
            candidate_set = tuple(sorted(candidate_ids))
            self.assertEqual(gold["candidate_ids"], list(candidate_ids))
            self.assertEqual(gold["target_position"], candidate_ids.index(gold["target_id"]))
            self.assertEqual(orders_by_set.setdefault(candidate_set, candidate_ids), candidate_ids)
            target_positions_by_set.setdefault(candidate_set, set()).add(gold["target_position"])
        self.assertTrue(all(positions == set(range(5)) for positions in target_positions_by_set.values()))

    def test_verifier_rejects_candidate_order_that_depends_on_target(self):
        ledgers = module.generate_ledgers(19, candidate_count=3)
        same_set_rows = {}
        for receiver, gold in zip(ledgers["receiver"], ledgers["gold"]):
            key = tuple(sorted(gold["candidate_ids"]))
            same_set_rows.setdefault(key, []).append((receiver, gold))
        pair = next(rows[:2] for rows in same_set_rows.values() if len(rows) >= 2)
        receiver, gold = pair[1]
        receiver["candidates"] = list(reversed(receiver["candidates"]))
        gold["candidate_ids"] = [row["candidate_id"] for row in receiver["candidates"]]
        gold["target_position"] = gold["candidate_ids"].index(gold["target_id"])
        with self.assertRaisesRegex(ValueError, "independent of the private target"):
            module.verify_ledgers(ledgers)

    def test_role_ledgers_do_not_expose_answer_key_to_receiver(self):
        ledgers = module.generate_ledgers(23, candidate_count=4)
        for sender, receiver, gold in zip(ledgers["sender"], ledgers["receiver"], ledgers["gold"]):
            self.assertEqual(sender["private_target_id"], gold["target_id"])
            self.assertNotIn("private_target_id", receiver)
            self.assertNotIn("target_id", receiver)
            self.assertNotIn("private_target", receiver)
            self.assertEqual(len(receiver["candidates"]), 4)
            self.assertIn(gold["target_id"], [row["candidate_id"] for row in receiver["candidates"]])
        module.verify_ledgers(ledgers)

    def test_all_candidate_counts_have_exact_uniform_no_message_posterior(self):
        for candidate_count in (2, 3, 5, 9):
            ledgers = module.generate_ledgers(4, candidate_count)
            self.assertEqual(ledgers["episode_count"], candidate_count * __import__("math").comb(9, candidate_count))
            self.assertAlmostEqual(ledgers["no_message_bayes_accuracy"], 1 / candidate_count)
            self.assertEqual(ledgers["ideal_zero_error_payload_floor_bits"], 4)

    def test_verifier_rejects_target_leakage_and_wrong_choice_count(self):
        ledgers = module.generate_ledgers(17)
        ledgers["receiver"][0]["private_target_id"] = ledgers["gold"][0]["target_id"]
        with self.assertRaisesRegex(ValueError, "leaks"):
            module.verify_ledgers(ledgers)

        ledgers = module.generate_ledgers(17)
        ledgers["receiver"][0]["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "match candidate_count"):
            module.verify_ledgers(ledgers)

    def test_writer_keeps_receiver_and_gold_in_separate_files(self):
        ledgers = module.generate_ledgers(2, candidate_count=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            module.write_ledgers(ledgers, Path(temp_dir))
            receiver_row = json.loads((Path(temp_dir) / "receiver.jsonl").read_text(encoding="utf-8").splitlines()[0])
            gold_row = json.loads((Path(temp_dir) / "gold.jsonl").read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("target_id", receiver_row)
        self.assertIn("target_id", gold_row)
        self.assertEqual(manifest["episode_count"], len(ledgers["gold"]))

    def test_invalid_seed_and_candidate_count_are_rejected(self):
        for seed, count in ((True, 5), (-1, 5), (0, 1), (0, 10), (0, True)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                module.generate_ledgers(seed, count)


if __name__ == "__main__":
    unittest.main()
