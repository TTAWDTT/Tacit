import unittest

from research.multiparty_sum_scaling import no_message_exact_success, scaling_table, scaling_row


class MultipartySumScalingTests(unittest.TestCase):
    def test_exact_binary_lower_bound_scales_linearly_and_is_attainable(self):
        rows = scaling_table([1, 2, 5, 10], domain_size=8)
        self.assertEqual(
            [row["exact_zero_error_payload_lower_bound_bits"] for row in rows],
            [3, 6, 15, 30],
        )
        self.assertEqual(
            [row["attainable_fixed_width_payload_bits"] for row in rows],
            [3, 6, 15, 30],
        )

    def test_no_message_exact_sum_mode(self):
        self.assertEqual(no_message_exact_success(1, 4), (1, 4))
        self.assertEqual(no_message_exact_success(2, 4), (4, 16))
        self.assertEqual(no_message_exact_success(3, 4), (12, 64))

    def test_messages_and_optional_llm_calls_are_distinct_counts(self):
        row = scaling_row(4, 8)
        self.assertEqual(row["simultaneous_messages"], 4)
        self.assertEqual(row["llm_calls_if_each_sender_and_referee_is_called_once"], 5)

    def test_rejects_non_power_of_two_domain_and_invalid_agent_count(self):
        with self.assertRaises(ValueError):
            scaling_row(2, 3)
        with self.assertRaises(ValueError):
            scaling_row(0, 4)


if __name__ == "__main__":
    unittest.main()
