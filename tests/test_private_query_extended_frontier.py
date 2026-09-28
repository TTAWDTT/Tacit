import unittest
from fractions import Fraction

from experiments.private_query_v0_4.extend_frontier import exact_frontier_cell


class ExtendedPrivateQueryFrontierTests(unittest.TestCase):
    def test_n5_one_bit_uniform_cell_matches_known_rac_law(self):
        row = exact_frontier_cell(5, 1, (1, 1, 1, 1, 1), (7, 1, 1, 1, 1))
        self.assertEqual(Fraction(row["success_fraction"]), Fraction(11, 16))
        self.assertEqual(row["codebooks_enumerated"], 496)

    def test_n5_two_bit_uniform_frontier_is_exact(self):
        row = exact_frontier_cell(5, 2, (1, 1, 1, 1, 1), (7, 1, 1, 1, 1))
        self.assertEqual(Fraction(row["success_fraction"]), Fraction(31, 40))
        self.assertEqual(row["codebooks_enumerated"], 35960)
        transfer = row["frozen_optimal_training_protocol_transfer"]
        self.assertLess(Fraction(transfer["success_min_fraction"]), Fraction(transfer["success_max_fraction"]))

    def test_skew_prior_changes_the_one_bit_optimum_and_transfer(self):
        skewed = exact_frontier_cell(5, 1, (7, 1, 1, 1, 1), (1, 1, 1, 1, 1))
        self.assertEqual(Fraction(skewed["success_fraction"]), Fraction(9, 11))
        self.assertEqual(
            Fraction(skewed["frozen_optimal_training_protocol_transfer"]["success_mean_fraction"]),
            Fraction(3, 5),
        )

    def test_rejects_unregistered_scale_or_budget(self):
        with self.assertRaises(ValueError):
            exact_frontier_cell(6, 1, (1,) * 6, (1,) * 6)
        with self.assertRaises(ValueError):
            exact_frontier_cell(5, 3, (1,) * 5, (1,) * 5)


if __name__ == "__main__":
    unittest.main()
