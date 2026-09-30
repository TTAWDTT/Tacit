import unittest

from research.prompt_search_selection_bias import expected_selection_optimism


class PromptSearchSelectionBiasTests(unittest.TestCase):
    def test_one_candidate_has_zero_selection_optimism(self):
        self.assertAlmostEqual(expected_selection_optimism(12, 0.4, 1), 0.0)

    def test_more_candidates_cannot_reduce_expected_selected_score(self):
        four = expected_selection_optimism(16, 0.5, 4)
        eight = expected_selection_optimism(16, 0.5, 8)
        self.assertGreater(eight, four)

    def test_reference_value(self):
        self.assertAlmostEqual(
            expected_selection_optimism(64, 0.5, 4), 0.064, delta=0.001
        )

    def test_rejects_invalid_domain(self):
        for args in (
            (0, 0.5, 4),
            (8, -0.1, 4),
            (8, float("nan"), 4),
            (8, 0.5, 0),
            (8.5, 0.5, 4),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                expected_selection_optimism(*args)


if __name__ == "__main__":
    unittest.main()
