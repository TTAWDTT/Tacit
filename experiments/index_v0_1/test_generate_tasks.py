"""Offline integrity tests for the INDEX_m task generator."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "index_v0_1"))
from generate_tasks import generate_episode, generate_shard, parse_exact_bit, score_episode


class IndexTaskTest(unittest.TestCase):
    def test_regeneration_is_deterministic(self):
        first = generate_shard("pilot", 77, [8, 16], 3)
        second = generate_shard("pilot", 77, [8, 16], 3)
        self.assertEqual(first, second)

    def test_split_streams_differ(self):
        train = generate_episode(77, "train", 16, 0)
        test = generate_episode(77, "test", 16, 0)
        self.assertNotEqual(train["sender_view"]["bits"], test["sender_view"]["bits"])
        self.assertNotEqual(train["receiver_view"]["index_1_based"], test["receiver_view"]["index_1_based"])

    def test_views_do_not_leak_the_other_input_or_answer(self):
        episode = generate_episode(20260928, "pilot", 32, 4)
        self.assertEqual(set(episode["sender_view"]), {"bits"})
        self.assertEqual(set(episode["receiver_view"]), {"index_1_based"})
        self.assertEqual(episode["gold_bit"], episode["sender_view"]["bits"][episode["receiver_view"]["index_1_based"] - 1])

    def test_strict_bit_scoring(self):
        self.assertEqual(parse_exact_bit(" 1\n"), 1)
        self.assertIsNone(parse_exact_bit("The answer is 1"))
        self.assertIsNone(parse_exact_bit("10"))
        episode = generate_episode(8, "pilot", 4, 0)
        answer = str(episode["gold_bit"])
        self.assertTrue(score_episode(episode, answer)["exact"])

    def test_invalid_task_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_episode(1, "pilot", 0, 0)
        with self.assertRaises(ValueError):
            generate_shard("pilot", 1, [8, 8], 2)


if __name__ == "__main__":
    unittest.main()
