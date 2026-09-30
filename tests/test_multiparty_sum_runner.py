from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import uuid

import pytest

from examples.multiparty_private_sum import _decode_sender_message
from experiments.multiparty_sum_v0_1.generate_tasks import ROOT, generate_dataset
from experiments.multiparty_sum_v0_1.runner import run_bundle_batch
from tacit import ChatCompletion


KEY = bytes(range(32))


class FakeAgent:
    def __init__(self, message_format: str = "decimal") -> None:
        self.message_format = message_format
        self.calls: list[dict] = []

    def complete(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        context = payload["private_context"]
        private = re.search(r"Your private integer is ([0-3])\.", context)
        if private:
            value = int(private.group(1))
            return ChatCompletion(str(value), "fake-sender")
        full = re.search(r"All private integers, in sender order: ([0-3](?:, [0-3])*)", context)
        if full:
            return ChatCompletion(str(sum(map(int, full.group(1).split(", ")))), "fake-receiver")
        count = int(re.search(r"sender_count=(\d+)", context).group(1))
        values = [
            _decode_sender_message(self.message_format, row["message"])
            for row in payload["visible_transcript"]
        ]
        known = [value for value in values if value is not None]
        answer = sum(known) + (3 * (count - len(known))) // 2
        return ChatCompletion(str(answer), "fake-receiver")


def _bundle_dir() -> Path:
    path = ROOT / ".cache" / f"test-multiparty-sum-runner-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_frozen_communicating_episode_uses_role_context_and_scores_gold() -> None:
    root = _bundle_dir()
    try:
        bundle = root / "bundle"
        generate_dataset(bundle, task_key=KEY, task_seed=3, agent_counts=(2,), episodes_per_count=2)
        senders, receiver = [FakeAgent(), FakeAgent()], FakeAgent()
        report = run_bundle_batch(
            bundle, agent_count=2, episode_indices=(0,), sender_clients=senders,
            receiver_client=receiver, request_cap=3,
        )
        episode = report["episodes"][0]
        assert report["planned_model_calls"] == report["actual_model_calls"] == 3
        assert episode["exact_success"] is True
        assert ["Your private integer is " in agent.calls[0]["private_context"] for agent in senders] == [True, True]
        assert "private integer is" not in receiver.calls[0]["private_context"]
        assert episode["episode_id"].startswith("sum-m02-")
    finally:
        shutil.rmtree(root)


def test_batch_call_cap_is_checked_before_any_fake_request() -> None:
    root = _bundle_dir()
    try:
        bundle = root / "bundle"
        generate_dataset(bundle, task_key=KEY, agent_counts=(2,), episodes_per_count=3)
        senders, receiver = [FakeAgent(), FakeAgent()], FakeAgent()
        with pytest.raises(ValueError, match="exceeding request cap"):
            run_bundle_batch(
                bundle, agent_count=2, episode_indices=(0, 1), sender_clients=senders,
                receiver_client=receiver, condition="communicate", request_cap=5,
            )
        assert not any(agent.calls for agent in [*senders, receiver])
    finally:
        shutil.rmtree(root)


def test_no_message_batch_can_cover_multiple_paired_rows_under_cap() -> None:
    root = _bundle_dir()
    try:
        bundle = root / "bundle"
        generate_dataset(bundle, task_key=KEY, agent_counts=(2,), episodes_per_count=3)
        senders, receiver = [FakeAgent(), FakeAgent()], FakeAgent()
        report = run_bundle_batch(
            bundle, agent_count=2, episode_indices=(0, 1, 2), sender_clients=senders,
            receiver_client=receiver, condition="no_message", request_cap=3,
        )
        assert report["actual_model_calls"] == 3
        assert all(row["model_calls"] == 1 for row in report["episodes"])
        assert len({row["episode_id"] for row in report["episodes"]}) == 3
        assert not any(agent.calls for agent in senders)
    finally:
        shutil.rmtree(root)
