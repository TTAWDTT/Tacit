"""Offline test that interrupted runs retain earlier completed votes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "hiddenbench_v0_8"))
from run_capability_screen import collect_agent_votes


class FakeAgent:
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response
        self.error = error

    def vote(self, prompt, possible_answers):
        if self.error:
            raise self.error
        return self.response


class CheckpointTest(unittest.TestCase):
    def test_previous_vote_survives_later_agent_failure(self):
        cache = ROOT / ".cache" / "tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary_dir:
            raw_path = Path(temporary_dir) / "partial.raw.json"
            record = {"initial_votes": []}
            agents = [
                FakeAgent("Agent 1", {"vote": "A", "reasoning": "private"}),
                FakeAgent("Agent 2", error=RuntimeError("simulated interruption")),
            ]
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                collect_agent_votes(agents, "vote prompt", ["A", "B"], record, raw_path)
            saved = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["initial_votes"]), 1)
            self.assertEqual(saved["initial_votes"][0]["agent"], "Agent 1")
            self.assertFalse(raw_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
