"""Generate seeded two-agent segmented prefix-sum tasks adapted from Silo II-11."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
SEED_BASE = 2026092701
LENGTHS = (6, 15, 30)
REPLICATES_PER_LENGTH = 4
VALUES = (1, 50)


def build_case(length: int, replicate: int) -> dict[str, Any]:
    seed = SEED_BASE + length * 100 + replicate
    rng = random.Random(seed)
    values = [rng.randint(*VALUES) for _ in range(2 * length)]
    shards = [values[:length], values[length:]]
    prefix: list[int] = []
    total = 0
    for value in values:
        total += value
        prefix.append(total)
    expected = [prefix[:length], prefix[length:]]
    case_id = f"PSUM-L{length:02d}-S{replicate:02d}"
    task = (
        "Compute the global inclusive prefix sum of the two concatenated private "
        "segments. Agent 0's segment comes first, followed by Agent 1's segment. "
        "Each agent must submit only the prefix sums corresponding to its own "
        "segment, as a JSON list of integers. Agent 1 needs the total of Agent "
        "0's segment before it can compute its first output."
    )
    agent_configs = []
    for agent_id, shard in enumerate(shards):
        agent_configs.append(
            {
                "agent_id": agent_id,
                "input_shard": shard,
                "user_prompt": (
                    f"Task: {task}\nYou are Agent {agent_id} of 2.\n"
                    f"Your private segment (in global order) is: {json.dumps(shard)}\n"
                    f"Your local segment length is {length}.\n"
                    "Use submit_result with your prefix-sum segment as a JSON list "
                    "of integers, in the same order and length as your local data."
                ),
            }
        )
    return {
        "case_id": case_id,
        "case_name": "Two-Agent Segmented Prefix Sum",
        "paradigm": "Silo-Bench II-11 Prefix Sum adaptation",
        "task_description": task,
        "metadata": {
            "num_agents": 2,
            "task_family": "prefix_sum",
            "segment_length": length,
            "value_range": list(VALUES),
            "replicate": replicate,
            "seed": seed,
            "split": "heldout",
            "communication_required_for_joint_success": True,
        },
        "agent_configs": agent_configs,
        "expected_output": {
            "type": "distributed",
            "per_agent_values": expected,
            "is_segmented": True,
            "verification_logic": "Each agent returns its exact global prefix-sum segment.",
        },
    }


def validate_case(case: dict[str, Any]) -> None:
    length = case["metadata"]["segment_length"]
    a, b = (agent["input_shard"] for agent in case["agent_configs"])
    expected_a, expected_b = case["expected_output"]["per_agent_values"]
    assert len(a) == len(b) == len(expected_a) == len(expected_b) == length
    local_prefix: list[int] = []
    total = 0
    for value in a:
        total += value
        local_prefix.append(total)
    assert local_prefix == expected_a
    for value in b:
        total += value
        local_prefix.append(total)
    assert local_prefix[length:] == expected_b
    # Agent 1's private data alone cannot determine its gold prefix vector:
    # changing Agent 0's segment changes every Agent 1 global prefix by a constant.
    alternate_a = [1] * length if sum(a) != length else [2] * length
    alternate_offset = sum(alternate_a)
    assert alternate_offset != sum(a)
    assert [alternate_offset + sum(b[: i + 1]) for i in range(length)] != expected_b


def write_suite() -> dict[str, Any]:
    TASKS.mkdir(parents=True, exist_ok=True)
    files = []
    for length in LENGTHS:
        for replicate in range(REPLICATES_PER_LENGTH):
            case = build_case(length, replicate)
            validate_case(case)
            path = TASKS / f"{case['case_id']}.json"
            content = json.dumps(case, ensure_ascii=False, indent=2) + "\n"
            path.write_text(content, encoding="utf-8", newline="\n")
            files.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "seed": case["metadata"]["seed"],
                    "segment_length": length,
                }
            )
    manifest = {
        "suite": "PrefixSum v0.1",
        "source_adaptation": "Silo-Bench II-11 Prefix Sum; Unlicense",
        "seed_base": SEED_BASE,
        "value_range": list(VALUES),
        "segment_lengths": list(LENGTHS),
        "replicates_per_length": REPLICATES_PER_LENGTH,
        "files": files,
    }
    manifest_path = TASKS / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate generated tasks without rewriting")
    args = parser.parse_args()
    if not args.check:
        write_suite()
    manifest = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = TASKS / item["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path.name}")
        validate_case(json.loads(content))
    print(f"Validated {len(manifest['files'])} held-out cases")


if __name__ == "__main__":
    main()
