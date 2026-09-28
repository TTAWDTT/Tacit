from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML/src"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_protocol_pilot_v0_6 import analyze, server_metrics
from hiddenbench.benchmark import load_benchmark
from protocol_engine import run_scenario


class FakeClient:
    model_name = "offline-fake"

    def chat(self, messages):
        return "Evidence is under review."

    def chat_json(self, messages, schema):
        return {"vote": schema["properties"]["vote"]["enum"][0], "rationale": "Offline test vote."}


class ProtocolAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        benchmark_path = SOURCE / "hiddenbench/data/benchmark_short.json"
        cls.task = next(task for task in load_benchmark(path=benchmark_path) if task.name == "evacuation_west_city")

    def test_server_metrics_filter_by_inclusive_task_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "I slot launch_slot_: id 0 | task 10 | processing task, is_child = 0\n"
                "I slot launch_slot_: id 0 | task 11 | processing task, is_child = 0\n"
                "I slot print_timing: id 0 | task 11 | prompt eval time = 12.0 ms / 4 tokens\n"
                "I slot print_timing: id 0 | task 11 | eval time = 34.0 ms / 8 tokens\n"
                "I slot print_timing: id 0 | task 11 | total time = 46.0 ms / 12 tokens\n"
                "I slot print_timing: id 0 | task 11 | n_tokens = 12, truncated = 0\n"
                "I slot stop processing: n_tokens = 12, truncated = 0 | task 11 | done\n"
                "I slot launch_slot_: id 0 | task 12 | processing task, is_child = 0\n",
                encoding="utf-8",
            )
            result = server_metrics(path, before=10, after=11)
            self.assertEqual(result["server_requests"], 1)
            self.assertEqual(result["server_completions"], 1)
            self.assertEqual(result["generated_tokens"], 8)
            self.assertEqual(result["prompt_eval_tokens_reported"], 4)
            self.assertEqual(result["summed_service_ms"], 46.0)

    def test_analyzer_scores_conditions_and_omits_private_task_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            server_log = root / "server.log"
            server_log.write_text("", encoding="utf-8")
            for index, protocol in enumerate(("natural_3", "exchange_decide", "reveal_all_3")):
                run = run_scenario(self.task, FakeClient(), "hidden", protocol, seed=20260927)
                payload = {
                    "metadata": {
                        "protocol": protocol,
                        "server_task_id_before": None,
                        "server_task_id_after": None,
                    },
                    "runs": [run],
                }
                (raw_dir / f"{protocol}.result.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            output = root / "sanitized.json"
            report = analyze(raw_dir, server_log, output, tokenize_fn=lambda text: len(text.split()))
            self.assertEqual([row["protocol"] for row in report["conditions"]], [
                "natural_3", "exchange_decide", "reveal_all_3"
            ])
            self.assertTrue(all(row["messages"] == 12 for row in report["conditions"]))
            sanitized = output.read_text(encoding="utf-8")
            for private_fact in self.task.hidden_information:
                self.assertNotIn(private_fact, sanitized)


if __name__ == "__main__":
    unittest.main()
