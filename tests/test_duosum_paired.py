"""Small deterministic checks for episode-clustered DuoSum analysis."""

from __future__ import annotations

import random
import unittest

from research.analyze_duosum_paired import CONDITIONS, _paired_delta, _sanitize


def _row(task_id: str, strict: int, payload_bytes: int) -> dict:
    return {
        "task_id": task_id,
        "strict_correct_agents": strict,
        "payload_bytes": payload_bytes,
    }


class DuoSumPairedTests(unittest.TestCase):
    def test_sanitizer_publishes_aggregate_fields_only(self) -> None:
        raw = []
        for version in ("pilot_v0_5", "pilot_v0_6"):
            version_rows = []
            for task_index in range(8):
                for condition in CONDITIONS:
                    version_rows.append({
                        "task": f"DUOSUM-B04-S{task_index:02}.json",
                        "condition": condition,
                        "strict_correct": 2,
                        "semantic_correct": 2,
                        "model_revision": "test-model",
                        "message_payload_bytes": 5,
                        "simulator_message_file_bytes": 13,
                        "total_tokens": 100,
                        "message_count": 2,
                        "elapsed_seconds": 1.0,
                        "submissions": [{"answer": "private-answer"}],
                        "case_dir": "private-path",
                    })
            raw.append(version_rows)
        result = _sanitize({"pilot_v0_5": raw[0], "pilot_v0_6": raw[1]})
        self.assertEqual(len(result), 2 * 8 * len(CONDITIONS))
        self.assertNotIn("submissions", result[0])
        self.assertNotIn("case_dir", result[0])
        self.assertTrue(result[0]["joint_strict_success"])

    def test_delta_pairs_by_task_and_uses_episode_cluster_metric(self) -> None:
        compact = [_row("a", 2, 4), _row("b", 1, 6)]
        baseline = [_row("b", 0, 8), _row("a", 0, 10)]
        point, low, high = _paired_delta(
            compact, baseline, "strict_agent_rate", random.Random(4)
        )
        self.assertEqual(point, 0.75)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_delta_rejects_unmatched_or_duplicate_tasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical task coverage"):
            _paired_delta(
                [_row("a", 1, 3)], [_row("b", 0, 4)], "payload_bytes", random.Random(1)
            )
        with self.assertRaisesRegex(ValueError, "identical task coverage"):
            _paired_delta(
                [_row("a", 1, 3), _row("a", 0, 4)],
                [_row("a", 1, 3), _row("b", 0, 4)],
                "payload_bytes",
                random.Random(1),
            )


if __name__ == "__main__":
    unittest.main()
