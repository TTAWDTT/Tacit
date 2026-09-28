import unittest
from fractions import Fraction

from experiments.private_query_v0_5.noisy_frontier import exact_cell


class PrivateQueryNoisyFrontierTests(unittest.TestCase):
    def test_noiseless_n4_matches_exact_frontier_reference(self):
        row = exact_cell(2, "0/1", "uniform", "skewed_first_coordinate_7_to_1")
        self.assertEqual(Fraction(row["training_prior_optimal_success_fraction"]), Fraction(13, 16))
        self.assertEqual(row["decoder_maps_enumerated"], 16**4)

    def test_half_noise_erases_all_transmitted_information(self):
        row = exact_cell(2, "1/2", "skewed_first_coordinate_7_to_1", "uniform")
        self.assertEqual(Fraction(row["training_prior_optimal_success_fraction"]), Fraction(1, 2))

    def test_budget_and_noise_levels_are_frozen(self):
        with self.assertRaises(ValueError):
            exact_cell(3, "1/8", "uniform", "uniform")
        with self.assertRaises(ValueError):
            exact_cell(1, "1/3", "uniform", "uniform")


if __name__ == "__main__":
    unittest.main()
