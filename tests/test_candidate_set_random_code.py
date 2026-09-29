from __future__ import annotations

import itertools
import unittest
from fractions import Fraction

from research.candidate_set_random_code import build_frontier, random_code_success


class CandidateSetRandomCodeTests(unittest.TestCase):
    def test_four_candidate_exact_values(self):
        self.assertEqual(random_code_success(candidate_size=4, message_symbols=1), Fraction(1, 4))
        self.assertEqual(random_code_success(candidate_size=4, message_symbols=2), Fraction(15, 32))
        self.assertEqual(random_code_success(candidate_size=4, message_symbols=3), Fraction(65, 108))
        self.assertEqual(random_code_success(candidate_size=4, message_symbols=4), Fraction(175, 256))

    def test_formula_matches_exhaustive_ideal_codebooks_on_small_alphabets(self):
        for candidate_size in range(1, 5):
            for message_symbols in range(1, 4):
                accuracies = []
                for mapping in itertools.product(range(message_symbols), repeat=candidate_size):
                    occupied = len(set(mapping))
                    accuracies.append(Fraction(occupied, candidate_size))
                exhaustive_mean = sum(accuracies, Fraction()) / len(accuracies)
                self.assertEqual(
                    exhaustive_mean,
                    random_code_success(candidate_size=candidate_size, message_symbols=message_symbols),
                )

    def test_frontier_is_monotone_and_labeled_as_non_language_reference(self):
        report = build_frontier(candidate_size=4, max_bits=5)
        points = report["frontier"]
        self.assertEqual([point["payload_bits"] for point in points], list(range(6)))
        self.assertTrue(all(a["expected_exact_selection"] <= b["expected_exact_selection"] for a, b in zip(points, points[1:])))
        self.assertFalse(report["is_language_or_llm_result"])
        self.assertTrue(any("codebook/seed distribution" in item for item in report["limits"]))

    def test_rejects_invalid_parameters(self):
        for k, m in ((0, 1), (True, 1), (4, 0), (4, False)):
            with self.assertRaises(ValueError):
                random_code_success(candidate_size=k, message_symbols=m)
        with self.assertRaises(ValueError):
            build_frontier(candidate_size=4, max_bits=17)


if __name__ == "__main__":
    unittest.main()
