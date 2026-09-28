from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".cache/research/HiddenBench_ICML/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hiddenbench.benchmark import load_benchmark
from protocol_engine import run_scenario


class FakeClient:
    model_name = "offline-fake"

    def chat(self, messages):
        return "Evidence is still being evaluated."

    def chat_json(self, messages, schema):
        return {"vote": schema["properties"]["vote"]["enum"][0], "rationale": "Offline test vote."}


class ProtocolEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = load_benchmark(path=ROOT / ".cache/research/HiddenBench_ICML/src/hiddenbench/data/benchmark_short.json")
        cls.task = next(task for task in tasks if task.name == "evacuation_west_city")

    def run_condition(self, protocol):
        return run_scenario(self.task, FakeClient(), "hidden", protocol, seed=20260927)

    def test_all_protocols_preserve_three_round_and_vote_structure(self):
        for protocol in ("natural_3", "exchange_decide", "reveal_all_3"):
            with self.subTest(protocol=protocol):
                result = self.run_condition(protocol)
                self.assertEqual(len(result["rounds"]), 3)
                self.assertEqual([len(r["messages"]) for r in result["rounds"]], [4, 4, 4])
                self.assertEqual(len(result["initial_votes"]), 4)
                self.assertEqual(len(result["final_votes"]), 4)
                self.assertEqual(len(result["discussion_transcript"]), 12)

    def test_exchange_decide_instructions_are_round_specific(self):
        result = self.run_condition("exchange_decide")
        for round_result in result["rounds"][:2]:
            self.assertTrue(all("EXCHANGE PHASE" in item["prompt"] for item in round_result["messages"]))
        self.assertTrue(all("DECIDE PHASE" in item["prompt"] for item in result["rounds"][2]["messages"]))

    def test_natural_condition_has_no_structured_phase_prompt(self):
        result = self.run_condition("natural_3")
        self.assertTrue(all("EXCHANGE PHASE" not in item["prompt"] for r in result["rounds"] for item in r["messages"]))
        self.assertTrue(all("DECIDE PHASE" not in item["prompt"] for r in result["rounds"] for item in r["messages"]))

    def test_reveal_all_appends_only_the_senders_visible_facts(self):
        result = self.run_condition("reveal_all_3")
        assignments = result["fact_assignments"]
        first_round = result["rounds"][0]["messages"]
        self.assertEqual(len(first_round), len(assignments))
        for index, message in enumerate(first_round):
            disclosed = "\n".join(f"- {fact}" for fact in assignments[index]["visible_facts"])
            self.assertIn("Information disclosed verbatim from this agent's profile:\n" + disclosed, message["response"])
        for round_result in result["rounds"][1:]:
            self.assertTrue(all("Information disclosed verbatim" not in item["response"] for item in round_result["messages"]))

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_condition("invented_code")


if __name__ == "__main__":
    unittest.main()
