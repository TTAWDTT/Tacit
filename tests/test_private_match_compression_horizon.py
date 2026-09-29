from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "private_match_v0_1" / "compression_horizon_sweep.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("private_match_compression_horizon_v0_1", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class CompressionHorizonSweepTests(unittest.TestCase):
    def test_small_sweep_is_deterministic_and_accounts_setup_once(self):
        kwargs = {"seeds": (12000, 22000), "horizons": (1, 4, 8)}
        first = module.run_sweep(**kwargs)
        second = module.run_sweep(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["dictionary"]["bytes"], 198)
        self.assertEqual(len(first["seed_results"]), 2)
        self.assertEqual(len(first["pooled_results"]), 12)
        for seed_result in first["seed_results"]:
            self.assertEqual(len(seed_result["horizons"]), 12)
            for row in seed_result["horizons"]:
                self.assertEqual(
                    row["dictionary_total_bytes"],
                    row["dictionary_setup_bytes"] + row["dictionary_stream_bytes"],
                )
                self.assertEqual(
                    row["net_savings_bytes_vs_no_dictionary"],
                    row["no_dictionary_stream_bytes"] - row["dictionary_total_bytes"],
                )
                self.assertFalse(row["dictionary_wins"])

    def test_stream_roundtrips_at_each_horizon(self):
        report = module.run_sweep(seeds=(12000,), horizons=(1, 8))
        self.assertTrue(all(row["no_dictionary_stream_bytes"] > 0 for row in report["seed_results"][0]["horizons"]))

    def test_non_increasing_horizons_are_rejected(self):
        with self.assertRaises(ValueError):
            module.run_sweep(seeds=(1,), horizons=(8, 4))


if __name__ == "__main__":
    unittest.main()
