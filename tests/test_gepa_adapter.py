from __future__ import annotations

import json
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from experiments.emergent_ood_v0_4 import episodes
from experiments.emergent_ood_v0_4.gepa_adapter import (
    RequestBudgetExceeded,
    BudgetedChatModel,
    RequestLedger,
    TacitGEPAAdapter,
    load_train_clusters,
    run_gepa_optimization,
)
from experiments.emergent_ood_v0_4.split import build_split
from tacit.runtime import ChatCompletion, OpenAICompatibleClient


ROOT = Path(__file__).resolve().parents[1]


class _TaskFake:
    model = "offline-fake"

    def __init__(self, role: str) -> None:
        self.role = role
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.role == "sender":
            payload = json.loads(messages[-1]["content"])
            private = json.loads(payload["private_context"])
            return ChatCompletion(json.dumps(private["private_meaning"]), self.model)
        payload = json.loads(messages[-1]["content"])
        receiver = json.loads(payload["private_context"])
        meaning = json.loads(payload["visible_transcript"][0]["message"])
        selected = next(
            item["candidate_id"] for item in receiver["candidates"]
            if item["attributes"] == meaning
        )
        return ChatCompletion(selected, self.model)


class _TextFake:
    model = "offline-text-fake"

    def complete(self, messages):
        return ChatCompletion("offline response", self.model, input_tokens=2, output_tokens=2, service_seconds=0.01)


class GepaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.directory = Path(self.temp.name)
        split = build_split(seed=71)
        bundle = episodes.generate_ledgers(
            split=split, task_key=bytes(range(32)), task_seed=9, k=2, sets_per_stage=3,
        )
        self.bundle_dir = self.directory / "episodes"
        episodes.write_ledgers(bundle, self.bundle_dir)
        calibration_split = build_split(seed=72)
        calibration_bundle = episodes.generate_ledgers(
            split=calibration_split, task_key=bytes(reversed(range(32))), task_seed=10, k=2, sets_per_stage=3,
        )
        self.calibration_dir = self.directory / "calibration"
        episodes.write_ledgers(calibration_bundle, self.calibration_dir)
        self.run_root = ROOT / ".cache" / "emergent_ood_v0_4" / "gepa_runs"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.run_temp = tempfile.TemporaryDirectory(dir=self.run_root)
        self.run_dir = Path(self.run_temp.name) / "run"

    def tearDown(self) -> None:
        self.run_temp.cleanup()
        self.temp.cleanup()

    def test_train_loader_does_not_open_or_verify_held_out_ledgers(self) -> None:
        (self.bundle_dir / "sender_validation.jsonl").write_text("deliberately damaged", encoding="utf-8")
        clusters, _split, _metadata = load_train_clusters(self.bundle_dir, split_seed=71)
        self.assertEqual(len(clusters), 3)
        self.assertEqual(len(clusters[0].episodes), 2)

    def test_fake_evaluation_scores_complete_cluster_and_enforces_global_cap(self) -> None:
        sender, receiver = _TaskFake("sender"), _TaskFake("receiver")
        adapter = TacitGEPAAdapter(
            episode_dir=self.bundle_dir, split_seed=71,
            sender_model=sender, receiver_model=receiver, request_budget=4,
        )
        candidate = {"sender_instruction": "Use exact attribute names and values.",
                     "receiver_instruction": "Match the full tuple exactly."}
        result = adapter.evaluate(adapter.clusters[:1], candidate, capture_traces=True)
        self.assertEqual(result.scores, [1.0], result.trajectories)
        self.assertEqual(adapter.ledger.used, 4)
        self.assertEqual(sender.calls, 2)
        self.assertEqual(receiver.calls, 2)
        traces = adapter.make_reflective_dataset(candidate, result, ["sender_instruction"])
        self.assertEqual(len(traces["sender_instruction"]), 2)

        with self.assertRaises(RequestBudgetExceeded):
            adapter.evaluate(adapter.clusters[1:], candidate)
        self.assertEqual(adapter.ledger.used, 4)

    def test_request_ledger_survives_adapter_checkpoint(self) -> None:
        adapter = TacitGEPAAdapter(
            episode_dir=self.bundle_dir, split_seed=71,
            sender_model=_TaskFake("sender"), receiver_model=_TaskFake("receiver"),
            request_budget=10,
        )
        adapter.sender_model.complete([{
            "role": "user",
            "content": json.dumps({"private_context": json.dumps({"private_meaning": {"a": "b"}})}),
        }])
        state = adapter.get_adapter_state()
        restored = TacitGEPAAdapter(
            episode_dir=self.bundle_dir, split_seed=71,
            sender_model=_TaskFake("sender"), receiver_model=_TaskFake("receiver"),
            request_budget=10,
        )
        restored.set_adapter_state(state)
        self.assertEqual(restored.ledger.used, 1)
        with self.assertRaises(ValueError):
            restored.set_adapter_state({**state, "limit": 9})

    def test_durable_ledger_counts_reflection_call_and_failure_across_restart(self) -> None:
        ledger_path = self.run_dir / "request-ledger.json"
        ledger = RequestLedger(2, path=ledger_path)
        reflection = BudgetedChatModel(_TextFake(), ledger, "reflection")
        self.assertEqual(reflection("reflect offline"), "offline response")
        self.assertEqual(reflection.batch_complete([[{"role": "user", "content": "one"}]]), ["offline response"])
        resumed = RequestLedger(2, path=ledger_path)
        self.assertEqual(resumed.used, 2)
        with self.assertRaises(RequestBudgetExceeded):
            BudgetedChatModel(_TextFake(), resumed, "sender").complete([])

        class _Failure:
            def complete(self, _messages):
                raise RuntimeError("offline failure")

        failure_path = self.directory / "failed-ledger.json"
        failing = RequestLedger(1, path=failure_path)
        with self.assertRaises(RuntimeError):
            BudgetedChatModel(_Failure(), failing, "reflection").complete([])
        restored_failure = RequestLedger(1, path=failure_path)
        self.assertEqual(restored_failure.used, 1)
        self.assertEqual(restored_failure.records[0].outcome, "failed")

    def test_optimizer_bridge_pins_api_and_keeps_gepa_validation_train_only(self) -> None:
        sender = OpenAICompatibleClient("http://127.0.0.1:8000/v1", "sender")
        receiver = OpenAICompatibleClient("http://127.0.0.1:8001/v1", "receiver")
        reflection = OpenAICompatibleClient("http://127.0.0.1:8002/v1", "reflector")
        adapter = TacitGEPAAdapter(
            episode_dir=self.bundle_dir, split_seed=71,
            sender_model=sender, receiver_model=receiver, request_budget=64,
            request_ledger_path=self.run_dir / "request-ledger.json",
            model_population_id="offline-fixture",
        )
        capability = self.directory / "capability.jsonl"
        capability.write_text("{}\n", encoding="utf-8")
        capability.with_suffix(".jsonl.manifest.json").write_text("{}", encoding="utf-8")
        preflight = self.directory / "preflight.json"
        preflight.write_text("{}", encoding="utf-8")
        refreshed_preflight = self.directory / "preflight-refreshed.json"
        refreshed_preflight.write_text('{"fresh": true}', encoding="utf-8")
        candidate = {"sender_instruction": "Use English attributes.",
                     "receiver_instruction": "Choose the matching identifier."}
        calls = {}

        def fake_optimize(**kwargs):
            calls.update(kwargs)
            self.assertIsNone(kwargs["valset"])
            self.assertIs(kwargs["adapter"], adapter)
            self.assertEqual(kwargs["trainset"], adapter.clusters)
            self.assertFalse(kwargs["cache_evaluation"])
            self.assertEqual(kwargs["max_metric_calls"], 9)
            self.assertEqual(kwargs["reflection_lm"]("offline reflection prompt"), "fake")
            return SimpleNamespace(
                best_candidate=candidate, best_idx=0,
                val_aggregate_scores=[0.75], num_candidates=2, total_metric_calls=17,
            )

        fake_gepa = ModuleType("gepa")
        fake_gepa.optimize = fake_optimize
        with (
            patch.dict(sys.modules, {"gepa": fake_gepa}),
            patch("experiments.emergent_ood_v0_4.gepa_adapter.importlib.metadata.version", return_value="0.1.4"),
            patch("experiments.emergent_ood_v0_4.gepa_adapter.validate_capability_ledger") as capability_gate,
            patch("experiments.emergent_ood_v0_4.gepa_adapter.validate_resource_preflight") as resource_gate,
            patch.object(OpenAICompatibleClient, "complete", return_value=ChatCompletion("fake", "offline", 2, 1, 0.01)) as complete,
        ):
            result, manifest = run_gepa_optimization(
                adapter,
                reflection_client=reflection,
                run_dir=self.run_dir,
                preflight_path=preflight,
                capability_episode_dir=self.calibration_dir,
                capability_split_seed=72,
                capability_ledger_path=capability,
                receiver_tokenizer_id="receiver-tok",
                sender_tokenizer_id="sender-tok",
                reflection_tokenizer_id="reflection-tok",
                max_metric_calls=9,
                seed=5,
                seed_candidate=candidate,
            )
            _resumed_result, resumed_manifest = run_gepa_optimization(
                adapter,
                reflection_client=reflection,
                run_dir=self.run_dir,
                preflight_path=refreshed_preflight,
                capability_episode_dir=self.calibration_dir,
                capability_split_seed=72,
                capability_ledger_path=capability,
                receiver_tokenizer_id="receiver-tok",
                sender_tokenizer_id="sender-tok",
                reflection_tokenizer_id="reflection-tok",
                max_metric_calls=9,
                seed=5,
                seed_candidate=candidate,
                resume=True,
            )
        self.assertEqual(result.best_idx, 0)
        self.assertEqual(manifest["best_training_mean_exact_selection"], 0.75)
        self.assertEqual(manifest["validation_rows_opened"], 0)
        self.assertEqual(manifest["test_rows_opened"], 0)
        capability_gate.assert_called_with(
            capability.resolve(),
            expected_calibration_episodes=ANY,
            evaluation_episodes=ANY,
            input_manifest_sha256=ANY,
            split_seed=72,
            split_sha256=ANY,
            evaluation_split_seed=71,
            task_seed=10,
            task_key_id=ANY,
            receiver_model="receiver",
            receiver_tokenizer_id="receiver-tok",
            model_population_id="offline-fixture",
            task_id=ANY,
            train_target_support_size=192,
            ontology_id=None,
        )
        self.assertEqual(capability_gate.call_count, 2)
        self.assertEqual(resource_gate.call_count, 2)
        self.assertEqual(resource_gate.call_args.kwargs["required_ports"], {8000, 8001, 8002})
        self.assertEqual(adapter.ledger.records[0].role, "reflection")
        self.assertEqual(adapter.ledger.records[0].outcome, "completed")
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(len(resumed_manifest["preflight_sha256s"]), 2)
        self.assertNotEqual(manifest["preflight_sha256"], resumed_manifest["preflight_sha256"])
        self.assertTrue((self.run_dir / "best-training-card.json").is_file())
        self.assertTrue((self.run_dir / "result-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
