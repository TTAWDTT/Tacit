"""Offline integrity tests for the pointer-chasing task generator and scorer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pointer_chasing_v0_1"))
from generate_tasks import (  # noqa: E402
    decode_pointer,
    encode_pointer,
    exact_no_message_diagnostic,
    generate_episode,
    generate_shard,
    oracle_relay,
    oracle_answer,
    parse_exact_bit,
    pointer_trace,
    score_episode,
    validate_episode,
)
from protocol_baselines import full_map_exchange, oracle_frontier_point  # noqa: E402


class PointerChasingTaskTest(unittest.TestCase):
    def test_regeneration_is_deterministic(self):
        first = generate_shard("pilot", 81, [4, 8], [2, 3], 3)
        second = generate_shard("pilot", 81, [4, 8], [2, 3], 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_hash_sampler_regression_vector(self):
        episode = generate_episode(123, "pilot", 4, 2, 0)
        self.assertEqual(episode["agent_a_view"]["function_values_1_based"], [2, 4, 3, 3])
        self.assertEqual(episode["agent_b_view"]["function_values_1_based"], [3, 2, 3, 3])
        self.assertEqual(episode["gold_bit"], 0)

    def test_split_and_condition_streams_are_distinct(self):
        train = generate_episode(81, "train", 16, 3, 0)
        test = generate_episode(81, "test", 16, 3, 0)
        deeper = generate_episode(81, "train", 16, 4, 0)
        self.assertNotEqual(train["agent_a_view"], test["agent_a_view"])
        self.assertNotEqual(train["agent_b_view"], test["agent_b_view"])
        self.assertNotEqual(train["agent_a_view"], deeper["agent_a_view"])

    def test_hand_computed_pointer_trace(self):
        function_a = [2, 3, 4, 1]
        function_b = [4, 3, 2, 1]
        self.assertEqual(pointer_trace(function_a, function_b, 3), [1, 2, 3, 4])
        self.assertEqual(pointer_trace(function_a, function_b, 4), [1, 2, 3, 4, 1])

    def test_fixed_width_pointer_codes_and_oracle_relay(self):
        for size in (2, 3, 4, 8, 9, 16):
            width = (size - 1).bit_length()
            for pointer in range(1, size + 1):
                message = encode_pointer(pointer, size)
                self.assertEqual(len(message), width)
                self.assertEqual(decode_pointer(message, size), pointer)
        episode = generate_episode(8, "pilot", 4, 3, 0)
        relay = oracle_relay(episode)
        self.assertEqual(len(relay["messages"]), episode["depth"])
        self.assertEqual(relay["payload_bit_count"], episode["depth"] * 2)
        self.assertEqual(relay["answer_bit"], oracle_answer(episode))
        self.assertFalse(relay["framing_included"])

    def test_full_map_exchange_is_exact_bilateral_one_batch_control(self):
        episode = generate_episode(1729, "pilot", 4, 3, 0)
        result = full_map_exchange(episode)
        self.assertTrue(result["both_agents_can_compute_answer"])
        self.assertEqual(result["answer_bit"], episode["gold_bit"])
        self.assertEqual(result["agent_answers"], {"agent_a": episode["gold_bit"], "agent_b": episode["gold_bit"]})
        self.assertEqual(result["synchronous_batches"], 1)
        self.assertEqual(result["directed_transmissions"], 2)
        self.assertEqual(result["aggregate_payload_bits"], 16)
        self.assertEqual(result["aggregate_payload_bytes_ascii"], 16)

    def test_oracle_frontier_uses_separate_round_and_bandwidth_axes(self):
        point = oracle_frontier_point(8, 3)
        self.assertEqual(point["pointer_relay"]["synchronous_batches"], 3)
        self.assertEqual(point["pointer_relay"]["aggregate_payload_bits"], 9)
        self.assertEqual(point["full_map_exchange"]["synchronous_batches"], 1)
        self.assertEqual(point["full_map_exchange"]["aggregate_payload_bits"], 48)

    def test_pointer_codec_rejects_invalid_codes(self):
        with self.assertRaises(ValueError):
            encode_pointer(0, 8)
        with self.assertRaises(ValueError):
            decode_pointer("111", 5)
        with self.assertRaises(ValueError):
            decode_pointer("00", 5)

    def test_role_views_exclude_other_input_and_answer(self):
        episode = generate_episode(20260929, "pilot", 32, 5, 9)
        self.assertEqual(set(episode["agent_a_view"]), {"function_values_1_based"})
        self.assertEqual(set(episode["agent_b_view"]), {"function_values_1_based"})
        self.assertNotIn("gold_bit", episode["agent_a_view"])
        self.assertNotIn("gold_bit", episode["agent_b_view"])
        self.assertEqual(episode["gold_bit"], oracle_answer(episode))

    def test_strict_bit_scoring(self):
        self.assertEqual(parse_exact_bit(" 1\n"), 1)
        self.assertIsNone(parse_exact_bit("answer: 1"))
        self.assertIsNone(parse_exact_bit("10"))
        episode = generate_episode(19, "pilot", 8, 3, 0)
        correct = str(episode["gold_bit"])
        wrong = str(1 - episode["gold_bit"])
        self.assertTrue(score_episode(episode, {"agent_a": correct, "agent_b": correct})["joint_exact"])
        one_wrong = score_episode(episode, {"agent_a": correct, "agent_b": wrong})
        self.assertTrue(one_wrong["exact_a"])
        self.assertFalse(one_wrong["exact_b"])
        self.assertFalse(one_wrong["joint_exact"])

    def test_depth_one_requires_sharing_the_answer(self):
        episode = generate_episode(28, "pilot", 8, 1, 0)
        correct = str(episode["gold_bit"])
        wrong = str(1 - episode["gold_bit"])
        score = score_episode(episode, {"agent_a": correct, "agent_b": wrong})
        self.assertTrue(score["exact_a"])
        self.assertFalse(score["joint_exact"])

    def test_exact_no_message_prior_for_n2_k1(self):
        result = exact_no_message_diagnostic(2, 1)
        self.assertEqual(result["episode_count"], 16)
        self.assertEqual(result["agent_a_individual_bayes_accuracy"], 1.0)
        self.assertEqual(result["agent_b_individual_bayes_accuracy"], 0.5)
        self.assertEqual(result["both_individual_map_predictions_joint_accuracy"], 0.5)
        self.assertEqual(result["joint_no_message_upper_bound"], 0.5)
        self.assertFalse(result["global_no_message_optimum_search_performed"])

    def test_n4_k2_has_exact_half_joint_no_message_upper_bound(self):
        result = exact_no_message_diagnostic(4, 2)
        self.assertEqual(result["joint_no_message_upper_bound"], 0.5)
        self.assertEqual(result["best_public_constant_accuracy"], 0.5)

    def test_exact_no_message_diagnostic_refuses_expensive_sizes(self):
        with self.assertRaises(ValueError):
            exact_no_message_diagnostic(6, 2)

    def test_malformed_episodes_and_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 1, 2, 0)
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 3, 2, 0)
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 8, 0, 0)
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 8, 2, -1)
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 8.0, 2, 0)
        with self.assertRaises(ValueError):
            generate_shard("pilot", 1, [8, 8], [2], 1)
        with self.assertRaises(ValueError):
            generate_shard("pilot", 1, [8, 9], [2], 1)
        episode = generate_episode(1, "pilot", 8, 2, 0)
        with self.assertRaises(ValueError):
            score_episode(episode, {"agent_a": str(episode["gold_bit"])})
        with self.assertRaises(ValueError):
            score_episode(episode, {"agent_a": 1, "agent_b": 1})
        episode = generate_episode(1, "pilot", 8, 2, 0)
        episode["agent_b_view"]["gold_bit"] = episode["gold_bit"]
        with self.assertRaises(ValueError):
            validate_episode(episode)


if __name__ == "__main__":
    unittest.main()
