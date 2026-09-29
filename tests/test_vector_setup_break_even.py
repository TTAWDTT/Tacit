import unittest
from fractions import Fraction

from research.vector_setup_break_even import vector_setup_break_even


class VectorSetupBreakEvenTests(unittest.TestCase):
    def test_horizon_is_the_slowest_saving_dimension(self):
        result = vector_setup_break_even(
            setup_cost={"wire_bytes": 100, "model_calls": 4},
            baseline_per_episode={"wire_bytes": 20, "model_calls": 5},
            candidate_per_episode={"wire_bytes": 10, "model_calls": 4},
        )
        self.assertTrue(result["finite_componentwise_break_even"])
        self.assertEqual(result["horizon_episodes"], 10)
        self.assertEqual(result["dimensions"]["wire_bytes"]["horizon_episodes"], 10)
        self.assertEqual(result["dimensions"]["model_calls"]["horizon_episodes"], 4)

    def test_positive_setup_and_equal_token_cost_never_break_even(self):
        result = vector_setup_break_even(
            setup_cost={"wire_bytes": 100, "tokenizer-A": 80},
            baseline_per_episode={"wire_bytes": 20, "tokenizer-A": 100},
            candidate_per_episode={"wire_bytes": 10, "tokenizer-A": 100},
        )
        self.assertFalse(result["finite_componentwise_break_even"])
        self.assertIsNone(result["horizon_episodes"])
        self.assertEqual(result["blocking_dimensions"], ["tokenizer-A"])

    def test_no_saving_dimension_blocks_even_without_setup(self):
        result = vector_setup_break_even(
            setup_cost={"latency": 0},
            baseline_per_episode={"latency": 1},
            candidate_per_episode={"latency": 2},
        )
        self.assertFalse(result["finite_componentwise_break_even"])
        self.assertEqual(result["dimensions"]["latency"]["status"],
                         "candidate_costs_more_per_episode")

    def test_fractional_costs_use_exact_ceiling(self):
        result = vector_setup_break_even(
            setup_cost={"bytes": Fraction(3, 2)},
            baseline_per_episode={"bytes": Fraction(5, 6)},
            candidate_per_episode={"bytes": Fraction(1, 3)},
        )
        self.assertEqual(result["horizon_episodes"], 3)

    def test_equal_zero_setup_dimension_allows_other_dimensions_to_cross(self):
        result = vector_setup_break_even(
            setup_cost={"bytes": 10, "tokens": 0},
            baseline_per_episode={"bytes": 4, "tokens": 8},
            candidate_per_episode={"bytes": 2, "tokens": 8},
        )
        self.assertTrue(result["finite_componentwise_break_even"])
        self.assertEqual(result["horizon_episodes"], 5)

    def test_rejects_incomplete_and_inexact_cost_vectors(self):
        with self.assertRaisesRegex(ValueError, "same non-empty dimensions"):
            vector_setup_break_even(
                setup_cost={"bytes": 1},
                baseline_per_episode={"tokens": 2},
                candidate_per_episode={"bytes": 1},
            )
        with self.assertRaisesRegex(ValueError, "rational"):
            vector_setup_break_even(
                setup_cost={"bytes": 1.5},
                baseline_per_episode={"bytes": 2},
                candidate_per_episode={"bytes": 1},
            )


if __name__ == "__main__":
    unittest.main()
