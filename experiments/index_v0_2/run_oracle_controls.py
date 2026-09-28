"""Run deterministic INDEX_m channel and scorer controls without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.index_v0_1 import generate_tasks as task_generator  # noqa: E402


PREREG_PATH = HERE / "preregistration.json"
EXPERIMENT_ID = "index-controls-v0.2"
SCHEMA_VERSION = "tlu.costs.v2"
CONDITIONS = (
    "no_message_fixed_zero",
    "single_agent_full_information_oracle",
    "one_way_full_vector_oracle",
    "interactive_index_query_oracle",
)
ORACLE_TOKENIZER = "not-applicable-oracle-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path: Path, prereg: dict[str, Any]) -> list[dict[str, Any]]:
    expected = prereg["task"]["manifest"]
    expected_sha = expected["sha256"]
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError(f"task manifest checksum mismatch: expected {expected_sha}, got {observed_sha}")

    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"task manifest line {line_number} is invalid JSON") from exc
            task_generator.validate_episode(task)
            tasks.append(task)

    if len(tasks) != expected["episode_count"]:
        raise ValueError(f"expected {expected['episode_count']} task episodes, found {len(tasks)}")
    expected_split = expected["split"]
    expected_seed = expected["master_seed"]
    expected_counts = {int(length): count for length, count in expected["episodes_per_length"].items()}
    observed_counts: dict[int, int] = {}
    for task in tasks:
        if task["split"] != expected_split or task["master_seed"] != expected_seed:
            raise ValueError("task manifest split or master seed differs from preregistration")
        observed_counts[task["length"]] = observed_counts.get(task["length"], 0) + 1
    if observed_counts != expected_counts:
        raise ValueError(f"task-length counts differ from preregistration: {observed_counts}")
    return tasks


def recipient_tokens(agent: str) -> dict[str, dict[str, Any]]:
    return {agent: {"tokenizer": ORACLE_TOKENIZER, "tokens": None}}


def protocol_ids(condition: str) -> dict[str, str]:
    return {
        "policy_id": condition,
        "code_id": {
            "no_message_fixed_zero": "no-channel-v1",
            "single_agent_full_information_oracle": "central-input-union-v1",
            "one_way_full_vector_oracle": "ascii-bit-vector-v1",
            "interactive_index_query_oracle": "ascii-fixed-width-index-plus-bit-v1",
        }[condition],
        "decoder_id": {
            "no_message_fixed_zero": "constant-zero-v1",
            "single_agent_full_information_oracle": "direct-index-v1",
            "one_way_full_vector_oracle": "one-based-vector-index-v1",
            "interactive_index_query_oracle": "fixed-width-query-v1",
        }[condition],
    }


def model_population(condition: str) -> tuple[str, dict[str, str]]:
    if condition == "no_message_fixed_zero":
        return "fixed-baseline-v1", {"receiver": "fixed-zero-v1"}
    if condition == "single_agent_full_information_oracle":
        return "centralized-oracle-v1", {"solver": "evaluator-oracle-v1"}
    return "distributed-oracle-v1", {
        "sender": "evaluator-oracle-v1",
        "receiver": "evaluator-oracle-v1",
    }


def transmission(round_number: int, sender: str, recipient: str, payload: str) -> dict[str, Any]:
    return {
        "round": round_number,
        "sender": sender,
        "recipients": [recipient],
        "payload_utf8_bytes": len(payload.encode("utf-8")),
        "framing_utf8_bytes": 0,
        "recipient_tokens": recipient_tokens(recipient),
    }


def run_condition(task: dict[str, Any], condition: str) -> dict[str, Any]:
    bits = task["sender_view"]["bits"]
    index = task["receiver_view"]["index_1_based"]
    gold = task["gold_bit"]
    transmitted: list[dict[str, Any]] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    if condition == "no_message_fixed_zero":
        prediction = 0
    elif condition == "single_agent_full_information_oracle":
        prediction = bits[index - 1]
    elif condition == "one_way_full_vector_oracle":
        payload = "".join(str(bit) for bit in bits)
        transmitted.append(transmission(1, "sender", "receiver", payload))
        receiver_bits = [int(character) for character in payload]
        prediction = receiver_bits[index - 1]
    elif condition == "interactive_index_query_oracle":
        width = math.ceil(math.log2(len(bits)))
        query = format(index - 1, f"0{width}b")
        transmitted.append(transmission(1, "receiver", "sender", query))
        queried_index = int(query, 2) if query else 0
        response = str(bits[queried_index])
        transmitted.append(transmission(2, "sender", "receiver", response))
        prediction = int(response)
    else:
        raise ValueError(f"unknown condition: {condition}")

    wall_seconds = time.perf_counter() - started_wall
    process_cpu_seconds = time.process_time() - started_cpu
    population_id, agent_models = model_population(condition)
    exact = prediction == gold
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": task["episode_id"],
        "stratum": {
            "experiment_id": EXPERIMENT_ID,
            "task_id": f"INDEX_m@{task['version']}",
            "split": task["split"],
            "task_parameters": {
                "m": task["length"],
                "vector_distribution": "iid_uniform_binary",
                "index_distribution": "uniform_1_based",
            },
            "model_population_id": population_id,
            "agent_models": agent_models,
            "scorer_id": "exact-bit-v1",
        },
        "protocol": protocol_ids(condition),
        "outcome": {"joint_success": exact, "answer_score": float(exact)},
        "transmissions": transmitted,
        "model_calls": [],
        "runtime": {
            "wall_seconds": wall_seconds,
            "tool_seconds": 0.0,
            "process_cpu_seconds": process_cpu_seconds,
            "process_gpu_seconds": None,
            "peak_rss_bytes": None,
            "peak_vram_bytes": None,
        },
        "setup": [],
    }


def run(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_condition(task, condition) for task in tasks for condition in CONDITIONS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path, help="frozen task JSONL matching the preregistered SHA-256")
    parser.add_argument("--output", required=True, type=Path, help="per-episode tlu.costs.v2 JSONL output")
    parser.add_argument("--preregistration", type=Path, default=PREREG_PATH)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("experiment_id") != EXPERIMENT_ID:
        parser.error("preregistration experiment_id mismatch")
    if args.tasks.resolve() == args.output.resolve():
        parser.error("task input and result output must be different paths")
    try:
        tasks = load_tasks(args.tasks, prereg)
        records = run(tasks)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"records": len(records), "conditions": list(CONDITIONS), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
