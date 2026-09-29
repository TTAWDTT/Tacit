from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "emergent_ood_v0_1" / "split.py"
SPEC = importlib.util.spec_from_file_location("emergent_ood_split_v0_1", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class EmergentOODSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_changes_with_seed(self):
        self.assertEqual(module.generate_split(17), module.generate_split(17))
        self.assertNotEqual(module.generate_split(17)["held_out_meanings"], module.generate_split(18)["held_out_meanings"])

    def test_split_has_18_train_and_9_held_out_and_verifies(self):
        split = module.generate_split(17)
        module.verify_split(split)
        self.assertEqual(split["counts"], {"universe": 27, "train": 18, "held_out": 9})

    def test_verifier_rejects_incomplete_partition(self):
        split = module.generate_split(17)
        split["train_meanings"].pop()
        with self.assertRaises(ValueError):
            module.verify_split(split)

    def test_negative_and_boolean_seeds_are_rejected(self):
        for seed in (-1, True):
            with self.assertRaises(ValueError):
                module.generate_split(seed)


if __name__ == "__main__":
    unittest.main()
