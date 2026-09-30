from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.emergent_ood_v0_4 import episodes
from experiments.emergent_ood_v0_4.gepa_adapter import (
    RequestBudgetExceeded,
    TacitGEPAAdapter,
    load_train_clusters,
)
from experiments.emergent_ood_v0_4.split import build_split
from tacit.runtime import ChatCompletion


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


class GepaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.directory = Path(self.temp.name)
        split = build_split(seed=71)
        bundle = episodes.generate_ledgers(
            split=split, task_key=bytes(range(32)), task_seed=9, k=2, sets_per_stage=2,
        )
        self.bundle_dir = self.directory / "episodes"
        episodes.write_ledgers(bundle, self.bundle_dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_train_loader_does_not_open_or_verify_held_out_ledgers(self) -> None:
        (self.bundle_dir / "sender_validation.jsonl").write_text("deliberately damaged", encoding="utf-8")
        clusters, _split = load_train_clusters(self.bundle_dir, split_seed=71)
        self.assertEqual(len(clusters), 2)
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


if __name__ == "__main__":
    unittest.main()
