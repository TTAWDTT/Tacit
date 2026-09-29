from __future__ import annotations

from itertools import combinations
import unittest

from experiments.condenseflow_audit_v0_1.counterexample import (
    fixed_compressor_counterexample,
    support_overlap_frontier,
)


class CondenseFlowBoundAuditTests(unittest.TestCase):
    def test_query_dependent_top_one_bound_does_not_cover_one_fixed_compressor(self):
        result = fixed_compressor_counterexample(5.0)
        self.assertGreater(result["attention_concentration_rho"], 0.9999)
        self.assertLess(result["theorem_bound"], 0.0002)
        self.assertGreater(result["minimum_error_over_one_fixed_row_stochastic_A"], 1.4)
        self.assertGreater(result["violation_ratio"], 10_000)
        self.assertLess(result["best_common_support_mass"], 0.0001)
        self.assertGreaterEqual(
            result["query_shared_support_bound"],
            result["minimum_error_over_one_fixed_row_stochastic_A"],
        )

    def test_larger_query_magnitude_sharpens_the_gap(self):
        moderate = fixed_compressor_counterexample(3.0)
        sharp = fixed_compressor_counterexample(5.0)
        self.assertGreater(sharp["violation_ratio"], moderate["violation_ratio"])

    def test_invalid_query_magnitude_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_compressor_counterexample(0.0)

    def test_distinct_salient_supports_control_fixed_capacity(self):
        rows = support_overlap_frontier(
            context_size=6,
            compression_slots=(1, 2, 4),
            distinct_salient_positions=(1, 2, 3, 6),
            diffuse_mass=0.06,
        )["rows"]
        by_setting = {
            (row["compression_slots"], row["distinct_salient_positions"]): row
            for row in rows
        }
        # Individual sparsity is unchanged as query supports spread out.
        self.assertEqual(
            by_setting[(2, 1)]["per_query_top_k_mass"],
            by_setting[(2, 6)]["per_query_top_k_mass"],
        )
        # A common two-slot summary stops covering all salient positions once
        # their count exceeds its capacity.
        self.assertGreater(by_setting[(2, 2)]["best_common_support_mass"], 0.9)
        self.assertLess(by_setting[(2, 3)]["best_common_support_mass"], 0.03)
        # Increasing capacity to cover all six restores shared mass.
        self.assertGreater(by_setting[(4, 3)]["best_common_support_mass"], 0.9)
        # Check the analytic frontier against exhaustive common-support search.
        for (slots, salient_count), row in by_setting.items():
            attention_rows = [
                [0.06 / 6 + (0.94 if position == salient else 0.0) for position in range(6)]
                for salient in range(salient_count)
            ]
            exact = max(
                min(sum(query[position] for position in support) for query in attention_rows)
                for support in combinations(range(6), slots)
            )
            self.assertAlmostEqual(row["best_common_support_mass"], exact)


if __name__ == "__main__":
    unittest.main()
