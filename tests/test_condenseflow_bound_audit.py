from __future__ import annotations

import unittest

from experiments.condenseflow_audit_v0_1.counterexample import fixed_compressor_counterexample


class CondenseFlowBoundAuditTests(unittest.TestCase):
    def test_query_dependent_top_one_bound_does_not_cover_one_fixed_compressor(self):
        result = fixed_compressor_counterexample(5.0)
        self.assertGreater(result["attention_concentration_rho"], 0.9999)
        self.assertLess(result["theorem_bound"], 0.0002)
        self.assertGreater(result["minimum_error_over_one_fixed_row_stochastic_A"], 1.4)
        self.assertGreater(result["violation_ratio"], 10_000)

    def test_larger_query_magnitude_sharpens_the_gap(self):
        moderate = fixed_compressor_counterexample(3.0)
        sharp = fixed_compressor_counterexample(5.0)
        self.assertGreater(sharp["violation_ratio"], moderate["violation_ratio"])

    def test_invalid_query_magnitude_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_compressor_counterexample(0.0)


if __name__ == "__main__":
    unittest.main()
