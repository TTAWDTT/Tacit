from __future__ import annotations

import unittest
from collections import Counter
from fractions import Fraction

from experiments.emergent_ood_v0_2.code_bounds import (
    as_dicts,
    fixed_width_lengths,
    framed_payload_codewords,
    huffman_prefix_lengths,
    summarize_uniform_codes,
)


class EmergentOODCodeBoundTests(unittest.TestCase):
    def test_nine_symbol_fixed_width_reference_is_four_bits(self):
        self.assertEqual(fixed_width_lengths(9), (4,) * 9)
        self.assertEqual(fixed_width_lengths(8), (3,) * 8)

    def test_nine_symbol_huffman_reference_is_seven_3_and_two_4(self):
        lengths = huffman_prefix_lengths(9)
        self.assertEqual(Counter(lengths), {3: 7, 4: 2})
        kraft_sum = sum(Fraction(1, 2**length) for length in lengths)
        self.assertEqual(kraft_sum, 1)
        self.assertEqual(Fraction(sum(lengths), len(lengths)), Fraction(29, 9))

    def test_external_frame_payload_reference_allows_prefix_related_strings(self):
        codewords = framed_payload_codewords(9)
        self.assertEqual(len(set(codewords)), 9)
        self.assertEqual(Counter(map(len, codewords)), {1: 2, 2: 4, 3: 3})
        self.assertEqual(sum(map(len, codewords)), 19)
        self.assertTrue(any(
            longer.startswith(shorter)
            for shorter in codewords
            for longer in codewords
            if shorter != longer
        ))

    def test_summary_does_not_claim_to_include_transport_framing(self):
        rows = summarize_uniform_codes(9)
        self.assertEqual([row.kind for row in rows], [
            "fixed_width", "prefix_free_huffman", "externally_framed_payload_only",
        ])
        self.assertTrue(all(not row.transport_framing_cost_included for row in rows))
        self.assertEqual([row.mean_bits for row in rows], [Fraction(4), Fraction(29, 9), Fraction(19, 9)])
        self.assertEqual([row.self_delimiting for row in rows], [True, True, False])
        self.assertEqual(as_dicts(9)[-1]["mean_payload_bits"], "19/9")

    def test_symbol_count_must_be_an_integer_at_least_two(self):
        for invalid in (True, 1, 0, -2, 2.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fixed_width_lengths(invalid)
                with self.assertRaises(ValueError):
                    huffman_prefix_lengths(invalid)
                with self.assertRaises(ValueError):
                    framed_payload_codewords(invalid)


if __name__ == "__main__":
    unittest.main()
