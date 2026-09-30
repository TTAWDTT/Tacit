from __future__ import annotations

import unittest

from research.audit_protocol_token_scaling import (
    CANDIDATES_PER_EPISODE,
    synthetic_episodes,
)


class ProtocolTokenScalingTests(unittest.TestCase):
    def test_synthetic_fixture_is_balanced_and_candidate_disjoint_by_meaning(self) -> None:
        episodes = synthetic_episodes(dimensions=3, values=2, episode_count=16)
        self.assertEqual(len(episodes), 16)
        gold_positions = [episode["gold_position"] for episode in episodes]
        self.assertEqual(gold_positions.count(0), 4)
        self.assertEqual(gold_positions.count(1), 4)
        self.assertEqual(gold_positions.count(2), 4)
        self.assertEqual(gold_positions.count(3), 4)

        for episode in episodes:
            target = episode["sender_target"]
            target_digits = tuple(
                int(target[f"axis{axis:02d}"].rsplit("_", 1)[1])
                for axis in range(3)
            )
            self.assertEqual(sum(target_digits) % 2, 0)
            candidates = episode["receiver_candidates"]
            self.assertEqual(len(candidates), CANDIDATES_PER_EPISODE)
            meanings = [candidate["attributes"] for candidate in candidates]
            self.assertEqual(len({tuple(sorted(row.items())) for row in meanings}), 4)
            self.assertEqual(meanings[episode["gold_position"]], target)

    def test_support_rank_mapping_is_bijective_for_small_modular_holdouts(self) -> None:
        from research.audit_protocol_token_scaling import _tuple_from_support_rank

        for dimensions, values in ((3, 2), (3, 4), (4, 3)):
            support = values ** (dimensions - 1)
            tuples = [
                _tuple_from_support_rank(rank, dimensions=dimensions, values=values)
                for rank in range(support)
            ]
            self.assertEqual(len(set(tuples)), support)
            self.assertTrue(all(sum(row) % values == 0 for row in tuples))

    def test_invalid_dimensions_or_unrepresentable_digit_alphabet_are_rejected(self) -> None:
        for dimensions, values in ((2, 4), (3, 1), (3, 11)):
            with self.subTest(dimensions=dimensions, values=values), self.assertRaises(ValueError):
                synthetic_episodes(dimensions=dimensions, values=values)


if __name__ == "__main__":
    unittest.main()
