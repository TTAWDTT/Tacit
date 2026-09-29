import math
import unittest

import numpy as np

from research.split_mean_power import (
    analyze,
    empirical_bernstein_radius,
    hoeffding_radius,
    scenario_power,
    validate_scenario,
)


class SplitMeanPowerTests(unittest.TestCase):
    def test_radius_matches_bounded_mean_hoeffding_bound(self):
        self.assertAlmostEqual(hoeffding_radius(clusters=100, alpha=0.05), math.sqrt(2 * math.log(40) / 100))

    def test_empirical_bernstein_radius_matches_transformed_paper_bound(self):
        effects = np.full(500, 0.1)
        expected = 14 * math.log(4 / 0.05) / (3 * (len(effects) - 1))
        self.assertAlmostEqual(empirical_bernstein_radius(split_effects=effects, alpha=0.05), expected)
        self.assertLess(expected, hoeffding_radius(clusters=len(effects), alpha=0.05))

    def test_empirical_bernstein_rejects_invalid_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            empirical_bernstein_radius(split_effects=np.array([0.1]), alpha=0.05)
        with self.assertRaisesRegex(ValueError, r"lie in \[-1, 1\]"):
            empirical_bernstein_radius(split_effects=np.array([0.1, 1.1]), alpha=0.05)

    def test_rejects_invalid_effect_support(self):
        with self.assertRaisesRegex(ValueError, r"outside \[-1, 1\]"):
            validate_scenario("bad", [{"difference": 1.1, "probability": 1.0}])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_scenario("bad", [{"difference": 0.0, "probability": 0.9}])

    def test_power_is_zero_when_interval_cannot_exclude_zero(self):
        result = scenario_power(
            support=[{"difference": 1.0, "probability": 1.0}],
            clusters=1,
            alpha=0.05,
            replicates=5,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result["rejections"], 0)
        self.assertEqual(result["estimated_power_or_type1"], 0.0)

    def test_empirical_bernstein_power_requires_two_independent_splits(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            scenario_power(
                support=[{"difference": 0.5, "probability": 1.0}],
                clusters=1,
                alpha=0.05,
                replicates=5,
                rng=np.random.default_rng(7),
                interval_method="empirical_bernstein",
            )

    def test_analysis_marks_single_split_empirical_interval_unavailable(self):
        result = analyze(
            scenarios={"positive": [{"difference": 0.25, "probability": 1.0}]},
            cluster_counts=(1,),
            replicates=10,
            seed=2,
        )
        row = result["scenarios"]["positive"]["operating_characteristics_by_interval"]["empirical_bernstein"][0]
        self.assertIsNone(row["estimated_power_or_type1"])
        self.assertIn("at least two", row["unavailable_reason"])

    def test_analysis_is_reproducible_and_labels_assumptions(self):
        scenario = {"positive": [{"difference": 0.25, "probability": 1.0}]}
        left = analyze(scenarios=scenario, cluster_counts=(8, 16), replicates=20, seed=3)
        right = analyze(scenarios=scenario, cluster_counts=(8, 16), replicates=20, seed=3)
        self.assertEqual(left, right)
        self.assertIn("not estimates from model data", left["interpretation"])
        self.assertEqual(left["scenarios"]["positive"]["assumed_mean_difference"], 0.25)
        self.assertIn("empirical_bernstein", left["scenarios"]["positive"]["operating_characteristics_by_interval"])


if __name__ == "__main__":
    unittest.main()
