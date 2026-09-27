"""Generate fresh short-shard PrefixSum tasks for role-capability calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
SEED_BASE = 2026110001
LENGTHS = (2, 3, 4)
REPLICATES_PER_LENGTH = 8
VALUES = (1, 50)


def prefix(values: list[int]) -> list[int]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def build_case(length: int, replicate: int) -> dict[str, Any]:
    seed = SEED_BASE + length * 100 + replicate
    rng = random.Random(seed)
    values = [rng.randint(*VALUES) for _ in range(2 * length)]
    shards = [values[:length], values[length:]]
    global_prefix = prefix(values)
    expected = [global_prefix[:length], global_prefix[length:]]
    case_id = f"PSUM2-L{length:02d}-S{replicate:02d}"
    task = (
        "Compute the global inclusive prefix sum of the two concatenated private "
        "segments. Agent 0's segment comes first, followed by Agent 1's segment. "
        "Each agent must submit only the prefix sums corresponding to its own "
        "segment, as a JSON list of integers. Agent 1 needs Agent 0's exact "
        "subtotal before it can compute its first output."
    )
    agents = []
    for agent_id, shard in enumerate(shards):
        role = (
            "ROLE: SENDER (Agent 0). Before submitting, compute the exact sum of "
            "your segment and send exactly one message to Agent 1 using the shared "
            "message representation condition. Then submit your own local inclusive "
            "prefix sums as a JSON list."
            if agent_id == 0
            else "ROLE: RECEIVER (Agent 1). Never send a message. Receive Agent 0's "
            "message before computing. Decode the shared message representation, "
            "add that offset to each local inclusive prefix sum, and submit the "
            "resulting global prefix segment as a JSON list."
        )
        agents.append(
            {
                "agent_id": agent_id,
                "input_shard": shard,
                "user_prompt": (
                    f"Task: {task}\nYou are Agent {agent_id} of 2.\n"
                    f"Your private segment (in global order) is: {json.dumps(shard)}\n"
                    f"Your local segment length is {length}.\n{role}\n"
                    "Return only the prefix sums for your own segment, in order, "
                    "using submit_result with a JSON list of integers."
                ),
            }
        )
    return {
        "case_id": case_id,
        "case_name": "Two-Agent Short-Shard Role-Explicit Prefix Sum",
        "paradigm": "Silo-Bench II-11 Prefix Sum short-shard capability calibration v0.3",
        "task_description": task,
        "metadata": {
            "num_agents": 2,
            "task_family": "prefix_sum",
            "task_version": "0.3-short-shards",
            "segment_length": length,
            "value_range": list(VALUES),
            "replicate": replicate,
            "seed": seed,
            "split": "heldout",
            "communication_required_for_joint_success": True,
        },
        "agent_configs": agents,
        "expected_output": {
            "type": "distributed",
            "per_agent_values": expected,
            "is_segmented": True,
            "verification_logic": "Each agent returns its exact global prefix-sum segment.",
        },
    }


def validate_case(case: dict[str, Any]) -> None:
    length = int(case["metadata"]["segment_length"])
    a, b = (agent["input_shard"] for agent in case["agent_configs"])
    expected_a, expected_b = case["expected_output"]["per_agent_values"]
    assert len(a) == len(b) == len(expected_a) == len(expected_b) == length
    assert expected_a == prefix(a)
    assert expected_b == [sum(a) + value for value in prefix(b)]
    alternate = [2] * length if sum(a) == length else [1] * length
    assert [sum(alternate) + value for value in prefix(b)] != expected_b
    assert case["metadata"]["split"] == "heldout"


def write_suite() -> dict[str, Any]:
    TASKS.mkdir(parents=True, exist_ok=True)
    files = []
    for length in LENGTHS:
        for replicate in range(REPLICATES_PER_LENGTH):
            case = build_case(length, replicate)
            validate_case(case)
            path = TASKS / f"{case['case_id']}.json"
            payload = json.dumps(case, ensure_ascii=False, indent=2) + "\n"
            path.write_text(payload, encoding="utf-8", newline="\n")
            files.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "seed": case["metadata"]["seed"],
                    "segment_length": length,
                }
            )
    manifest = {
        "suite": "PrefixSum short-shard calibration v0.3",
        "source_adaptation": "Silo-Bench II-11 Prefix Sum; Unlicense",
        "seed_base": SEED_BASE,
        "value_range": list(VALUES),
        "segment_lengths": list(LENGTHS),
        "replicates_per_length": REPLICATES_PER_LENGTH,
        "files": files,
    }
    (TASKS / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        write_suite()
    manifest = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = TASKS / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Task checksum mismatch: {path.name}")
        validate_case(json.loads(path.read_text(encoding="utf-8")))
    print(f"Validated {len(manifest['files'])} held-out short-shard cases")


if __name__ == "__main__":
    main()
