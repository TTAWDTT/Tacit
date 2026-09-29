from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.emergent_ood_v0_4.split import (
    build_split, build_split_from_spec, load_ontology_spec, split_task_id, validate_split,
)


class EmergentOODV04SplitTests(unittest.TestCase):
    def test_default_split_has_fourth_order_holdout_and_complete_lower_order_coverage(self):
        split = build_split(seed=17)
        self.assertEqual(split["universe_size"], 256)
        self.assertEqual(len(split["train_meaning_ids"]), 192)
        self.assertEqual(len(split["held_out_meaning_ids"]), 64)
        self.assertEqual(split["coverage_by_order"], {"1": 16, "2": 96, "3": 256})
        validate_split(split)

    def test_split_is_deterministic_and_seed_changes_partition(self):
        first = build_split(seed=41)
        self.assertEqual(first, build_split(seed=41))
        self.assertNotEqual(first["held_out_meaning_ids"], build_split(seed=42)["held_out_meaning_ids"])

    def test_custom_three_by_two_space_obeys_same_coverage_rule(self):
        split = build_split(
            seed=5,
            attributes=("a", "b", "c"),
            values=(("a0", "a1"), ("b0", "b1"), ("c0", "c1")),
        )
        self.assertEqual(split["universe_size"], 8)
        self.assertEqual(len(split["train_meaning_ids"]), 4)
        self.assertEqual(len(split["held_out_meaning_ids"]), 4)
        self.assertEqual(split["coverage_by_order"], {"1": 6, "2": 12})

    def test_distinct_ontology_specs_create_hash_separated_matched_structure(self):
        root = Path(__file__).resolve().parents[1]
        robotics = load_ontology_spec(root / "experiments/emergent_ood_v0_4/ontologies/robotics_v1.json")
        music = load_ontology_spec(root / "experiments/emergent_ood_v0_4/ontologies/music_v1.json")
        splits = []
        for ontology in (robotics, music):
            spec = {
                "seed": 13,
                "attributes": ontology["attributes"],
                "values_by_attribute": ontology["values_by_attribute"],
                "ontology_id": ontology["ontology_id"],
            }
            splits.append(build_split_from_spec(seed=13, spec=spec))
        first, second = splits
        self.assertEqual(first["universe_size"], second["universe_size"])
        self.assertEqual(first["coverage_by_order"], second["coverage_by_order"])
        self.assertEqual(len(first["held_out_meaning_ids"]), len(second["held_out_meaning_ids"]))
        self.assertNotEqual(first["split_sha256"], second["split_sha256"])
        self.assertNotEqual(split_task_id(first), split_task_id(second))
        validate_split(first)
        validate_split(second)
        with self.assertRaisesRegex(ValueError, "seed does not match"):
            build_split_from_spec(seed=14, spec={
                "seed": 13, "attributes": robotics["attributes"],
                "values_by_attribute": robotics["values_by_attribute"],
                "ontology_id": robotics["ontology_id"],
            })

    def test_validation_rejects_tampering(self):
        split = build_split(seed=3)
        corrupt = copy.deepcopy(split)
        corrupt["meanings"][0]["split"] = "held_out"
        with self.assertRaises(ValueError):
            validate_split(corrupt)
        duplicate_id = copy.deepcopy(split)
        duplicate_id["train_meaning_ids"].append(duplicate_id["train_meaning_ids"][0])
        with self.assertRaises(ValueError):
            validate_split(duplicate_id)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_split(seed=True)
        with self.assertRaises(ValueError):
            build_split(attributes=("same", "same"), values=(("a", "b"), ("c", "d")))
        with self.assertRaises(ValueError):
            build_split(attributes=("a", "b"), values=(("a", "b"), ("c",)))


if __name__ == "__main__":
    unittest.main()
