from __future__ import annotations

import itertools
import unittest
from math import factorial
from fractions import Fraction

from research.protocol_onboarding import (
    build_report,
    disjoint_holdout_candidate_accuracy,
    holistic_onboarding_accuracy,
)


class ProtocolOnboardingTests(unittest.TestCase):
    def test_examples_for_four_meaning_holistic_code(self):
        self.assertEqual(holistic_onboarding_accuracy(meaning_count=4, distinct_examples=0), Fraction(1, 4))
        self.assertEqual(holistic_onboarding_accuracy(meaning_count=4, distinct_examples=1), Fraction(1, 2))
        self.assertEqual(holistic_onboarding_accuracy(meaning_count=4, distinct_examples=2), Fraction(3, 4))
        self.assertEqual(holistic_onboarding_accuracy(meaning_count=4, distinct_examples=3), Fraction(1, 1))

    def test_formula_matches_exhaustive_random_bijections(self):
        for meaning_count in range(1, 6):
            for examples in range(meaning_count):
                permutation_scores = []
                for symbols in itertools.permutations(range(meaning_count)):
                    known_symbols = set(symbols[:examples])
                    correct_if_seen = sum(symbol in known_symbols for symbol in symbols)
                    unseen_meanings = meaning_count - examples
                    expected = Fraction(correct_if_seen, meaning_count)
                    if unseen_meanings:
                        expected += Fraction(unseen_meanings, meaning_count * unseen_meanings)
                    permutation_scores.append(expected)
                exhaustive_mean = sum(permutation_scores, Fraction()) / len(permutation_scores)
                self.assertEqual(
                    exhaustive_mean,
                    holistic_onboarding_accuracy(
                        meaning_count=meaning_count,
                        distinct_examples=examples,
                    ),
                )

    def test_report_marks_lookup_limit_and_is_monotone(self):
        report = build_report(meaning_count=5)
        accuracies = [row["expected_exact_accuracy"] for row in report["frontier"]]
        self.assertTrue(all(a <= b for a, b in zip(accuracies, accuracies[1:])))
        self.assertFalse(report["is_language_or_llm_result"])
        self.assertTrue(any("not compositional" in limit for limit in report["limits"]))

    def test_random_holistic_code_has_no_value_on_disjoint_candidate_support(self):
        self.assertEqual(disjoint_holdout_candidate_accuracy(candidate_count=1), Fraction(1, 1))
        self.assertEqual(disjoint_holdout_candidate_accuracy(candidate_count=4), Fraction(1, 4))
        with self.assertRaises(ValueError):
            disjoint_holdout_candidate_accuracy(candidate_count=0)

    def test_disjoint_support_null_matches_exhaustive_unseen_bijections(self):
        for meaning_count in range(2, 6):
            meanings = set(range(meaning_count))
            for calibration_size in range(meaning_count):
                for calibrated in itertools.combinations(sorted(meanings), calibration_size):
                    remaining = sorted(meanings - set(calibrated))
                    for candidate_count in range(1, len(remaining) + 1):
                        for candidates in itertools.combinations(remaining, candidate_count):
                            unseen_symbols = set(range(meaning_count - calibration_size))
                            assignment_count = factorial(len(remaining))
                            success_mass = Fraction()
                            all_assignments = list(itertools.permutations(sorted(unseen_symbols)))
                            for symbol in unseen_symbols:
                                best_count = max(
                                    sum(assignment[remaining.index(candidate)] == symbol for assignment in all_assignments)
                                    for candidate in candidates
                                )
                                success_mass += Fraction(best_count, candidate_count * assignment_count)
                            self.assertEqual(
                                success_mass,
                                disjoint_holdout_candidate_accuracy(candidate_count=candidate_count),
                            )

    def test_rejects_invalid_domains(self):
        for meaning_count, examples in ((0, 0), (True, 0), (4, -1), (4, 4), (4, True)):
            with self.assertRaises(ValueError):
                holistic_onboarding_accuracy(meaning_count=meaning_count, distinct_examples=examples)
        with self.assertRaises(ValueError):
            build_report(meaning_count=0)


if __name__ == "__main__":
    unittest.main()
