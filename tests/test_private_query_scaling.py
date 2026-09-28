"""Regression checks for the exact classical one-bit RAC reference curve."""

from __future__ import annotations

import unittest
from fractions import Fraction

from experiments.private_query_v0_3.exact_one_bit_scaling import (
    exact_success,
    make_report,
    selected_scales,
)


class PrivateQueryScalingTests(unittest.TestCase):
    def test_exact_formula_matches_exhaustive_small_instance_optima(self) -> None:
        expected = {
            1: Fraction(1, 1),
            2: Fraction(3, 4),
            3: Fraction(3, 4),
            4: Fraction(11, 16),
        }
        self.assertEqual({n: exact_success(n) for n in expected}, expected)

    def test_invalid_source_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "n must be positive"):
            exact_success(0)
        with self.assertRaisesRegex(ValueError, "max_n must be positive"):
            selected_scales(0)

    def test_scale_selection_includes_dense_small_range_and_requested_endpoint(self) -> None:
        scales = selected_scales(4096)
        self.assertEqual(scales[:128], list(range(1, 129)))
        self.assertIn(256, scales)
        self.assertIn(1024, scales)
        self.assertEqual(scales[-1], 4096)

    def test_report_is_monotone_and_tracks_asymptotic_approach(self) -> None:
        report = make_report(4096)
        rows = {row["n_source_bits"]: row for row in report["rows"]}
        probabilities = [rows[n]["success_probability"] for n in sorted(rows)]
        self.assertTrue(all(a >= b for a, b in zip(probabilities, probabilities[1:])))
        self.assertLess(rows[4096]["success_probability"], rows[128]["success_probability"])
        self.assertAlmostEqual(rows[4096]["exact_to_asymptotic_advantage_ratio"], 1.0, delta=0.001)
        self.assertEqual(report["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
