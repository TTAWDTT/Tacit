from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest

from experiments.multiparty_sum_v0_1.generate_tasks import (
    ROOT,
    generate_dataset,
    model_visible_view,
    validate_dataset,
)


KEY = bytes(range(32))


def _test_dir() -> Path:
    path = ROOT / ".cache" / f"test-multiparty-sum-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_bundle_is_deterministic_role_separated_and_hash_bound() -> None:
    root = _test_dir()
    try:
        out_a, out_b = root / "a", root / "b"
        manifest_a = generate_dataset(out_a, task_key=KEY, task_seed=7, agent_counts=(2, 4), episodes_per_count=5)
        manifest_b = generate_dataset(out_b, task_key=KEY, task_seed=7, agent_counts=(2, 4), episodes_per_count=5)
        assert manifest_a["files_sha256"] == manifest_b["files_sha256"]
        validate_dataset(out_a)

        receiver = [json.loads(line) for line in (out_a / "receiver_m04.jsonl").read_text().splitlines()]
        gold = [json.loads(line) for line in (out_a / "gold_m04.jsonl").read_text().splitlines()]
        senders = [
            [json.loads(line) for line in (out_a / f"sender_m04_s{i:02d}.jsonl").read_text().splitlines()]
            for i in range(1, 5)
        ]
        for j, (r, g) in enumerate(zip(receiver, gold)):
            assert "exact_sum" not in r and "values_in_sender_order" not in r
            assert "private_value" not in r
            values = [s[j]["private_value"] for s in senders]
            assert g["values_in_sender_order"] == values
            assert g["exact_sum"] == sum(values)
            assert len({s[j]["episode_id"] for s in senders} | {r["episode_id"], g["episode_id"]}) == 1
            visible = model_visible_view(r)
            assert "episode_id" not in visible
            assert visible["sender_count"] == 4 and visible["prior"] == r["prior"]
    finally:
        shutil.rmtree(root)


def test_seed_and_key_domain_separate_tuples() -> None:
    root = _test_dir()
    try:
        a, b, c = root / "a", root / "b", root / "c"
        generate_dataset(a, task_key=KEY, task_seed=1, agent_counts=(3,), episodes_per_count=4)
        generate_dataset(b, task_key=KEY, task_seed=2, agent_counts=(3,), episodes_per_count=4)
        generate_dataset(c, task_key=bytes(reversed(KEY)), task_seed=1, agent_counts=(3,), episodes_per_count=4)
        payload = lambda path: (path / "gold_m03.jsonl").read_bytes()
        assert payload(a) != payload(b)
        assert payload(a) != payload(c)
    finally:
        shutil.rmtree(root)


def test_episode_prefix_is_stable_when_sample_size_changes() -> None:
    root = _test_dir()
    try:
        short, long = root / "short", root / "long"
        generate_dataset(short, task_key=KEY, task_seed=19, agent_counts=(2, 5), episodes_per_count=4)
        generate_dataset(long, task_key=KEY, task_seed=19, agent_counts=(2, 5), episodes_per_count=9)
        for m in (2, 5):
            for filename in [f"gold_m{m:02d}.jsonl", *[f"sender_m{m:02d}_s{i:02d}.jsonl" for i in range(1, m + 1)]]:
                prefix = (short / filename).read_text(encoding="utf-8").splitlines()
                expanded = (long / filename).read_text(encoding="utf-8").splitlines()
                assert expanded[:4] == prefix
    finally:
        shutil.rmtree(root)


def test_overwrite_tamper_and_call_limit_are_rejected() -> None:
    root = _test_dir()
    try:
        out = root / "bundle"
        generate_dataset(out, task_key=KEY, agent_counts=(2,), episodes_per_count=1)
        with pytest.raises(FileExistsError):
            generate_dataset(out, task_key=KEY, agent_counts=(2,), episodes_per_count=1)
        (out / "receiver_m02.jsonl").write_text("{}\n")
        with pytest.raises(ValueError, match="hash mismatch"):
            validate_dataset(out)
        with pytest.raises(ValueError, match="call ceiling"):
            generate_dataset(root / "too_many", task_key=KEY, agent_counts=(12,), episodes_per_count=1)
    finally:
        shutil.rmtree(root)
