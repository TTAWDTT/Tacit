from __future__ import annotations

import unittest

from experiments.emergent_ood_v0_4.episodes import (
    generate_ledgers,
    verify_ledgers,
    write_ledgers,
)
from experiments.emergent_ood_v0_4.split import build_split


class EmergentOODV04EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.split = build_split(seed=17)
        self.key = bytes(range(32))

    def test_role_ledgers_are_balanced_and_hide_evaluator_labels(self):
        bundle = generate_ledgers(split=self.split, task_key=self.key, task_seed=3, k=4, sets_per_stage=3)
        verify_ledgers(bundle)
        self.assertEqual(bundle["manifest"]["partition_sizes"], {"train": 192, "validation": 16, "test": 48})
        self.assertEqual(bundle["manifest"]["episodes_per_stage"], {"train": 12, "validation": 12, "test": 12})
        for stage, rows in bundle["sender"].items():
            self.assertTrue(all(set(row) == {"private_meaning"} for row in rows))
            self.assertTrue(all(set(row) == {"candidates"} for row in bundle["receiver"][stage]))
            self.assertTrue(all("episode_id" in row and "meaning_id" in row for row in bundle["gold"][stage]))
        self.assertEqual(bundle["manifest"]["chance_accuracy"], 0.25)

    def test_key_and_seed_reproduce_ledgers_and_distinct_keys_change_them(self):
        first = generate_ledgers(split=self.split, task_key=self.key, task_seed=19, k=4, sets_per_stage=2)
        repeated = generate_ledgers(split=self.split, task_key=self.key, task_seed=19, k=4, sets_per_stage=2)
        other = generate_ledgers(split=self.split, task_key=b"x" * 32, task_seed=19, k=4, sets_per_stage=2)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["gold"], other["gold"])

    def test_episode_ids_are_disjoint_across_split_seeds(self):
        first = generate_ledgers(split=self.split, task_key=self.key, task_seed=19, k=4, sets_per_stage=2)
        other_split = build_split(seed=18)
        other = generate_ledgers(split=other_split, task_key=self.key, task_seed=19, k=4, sets_per_stage=2)
        first_ids = {row["episode_id"] for stage in first["gold"].values() for row in stage}
        other_ids = {row["episode_id"] for stage in other["gold"].values() for row in stage}
        self.assertFalse(first_ids & other_ids)

    def test_rejects_short_keys_bad_k_and_escape_output_path(self):
        with self.assertRaises(ValueError):
            generate_ledgers(split=self.split, task_key=b"short")
        with self.assertRaises(ValueError):
            generate_ledgers(split=self.split, task_key=self.key, k=1)
        with self.assertRaises(ValueError):
            write_ledgers({}, self.split_path_outside_project())

    @staticmethod
    def split_path_outside_project():
        from pathlib import Path
        return Path("../outside-tac-it-should-not-be-written")


if __name__ == "__main__":
    unittest.main()
