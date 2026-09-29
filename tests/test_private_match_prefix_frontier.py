import unittest
from fractions import Fraction

from experiments.private_match_v0_3.prefix_frontier import (
    optimal_prefix_partition,
    prefix_code_frontier,
)


class PrivateMatchPrefixFrontierTests(unittest.TestCase):
    def test_q4_one_sender_expected_length_by_number_of_classes(self):
        expected = ["0", "1", "3/2", "2"]
        for class_count, expected_bits in enumerate(expected, start=1):
            with self.subTest(class_count=class_count):
                self.assertEqual(
                    str(optimal_prefix_partition(4, class_count).expected_bits),
                    expected_bits,
                )

    def test_q4_three_class_encoder_decodes_to_one_representative_per_class(self):
        code = optimal_prefix_partition(4, 3)
        self.assertEqual(sorted(value for group in code.classes for value in group), [0, 1, 2, 3])
        for value in range(4):
            decoded = code.decode_representative(code.encode(value))
            self.assertIn(decoded, next(group for group in code.classes if value in group))
        self.assertEqual(code.expected_bits, Fraction(3, 2))

    def test_q4_expected_length_frontier_exposes_fractional_success_points(self):
        points = prefix_code_frontier(4)["expected_length_pareto_frontier"]
        self.assertEqual(
            [(point["expected_payload_bits"], point["joint_success"]) for point in points],
            [
                ("0", "1/16"),
                ("1", "1/8"),
                ("3/2", "3/16"),
                ("2", "1/4"),
                ("5/2", "3/8"),
                ("3", "9/16"),
                ("7/2", "3/4"),
                ("4", "1"),
            ],
        )

    def test_q4_three_bit_expected_rate_exceeds_fixed_width_three_bit_success(self):
        report = prefix_code_frontier(4)
        point = next(
            item for item in report["expected_length_pareto_frontier"]
            if item["expected_payload_bits"] == "3"
        )
        fixed = next(
            item for item in report["fixed_width_integer_budget_reference"]
            if item["total_worst_case_bits"] == 3
        )
        self.assertEqual(point["joint_success"], "9/16")
        self.assertEqual(fixed["joint_success"], "1/2")
        allocation = next(
            item for item in point["sender_class_allocations"]
            if item["sender_x_classes"] == 3 and item["sender_y_classes"] == 3
        )
        self.assertEqual(allocation["worst_case_payload_bits"], 4)

    def test_prefix_code_rejects_invalid_messages_and_oversized_search(self):
        code = optimal_prefix_partition(4, 3)
        with self.assertRaisesRegex(ValueError, "not a codeword"):
            code.decode_representative("111")
        with self.assertRaisesRegex(ValueError, "2..32"):
            prefix_code_frontier(33)


if __name__ == "__main__":
    unittest.main()
