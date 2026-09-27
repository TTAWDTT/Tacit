"""Run a small exploratory representation pilot on upstream Silo-Bench tasks.

Requires the local server from research/local_chat_server.py at 127.0.0.1:8000.
Upstream checkout and generated run data stay in .cache/ (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache" / "upstream-silo-bench"
TASKS = UPSTREAM / "benchmarks"
OUTPUT = ROOT / ".cache" / "pilot_v0_1"
POLICIES = json.loads(
    (ROOT / "experiments" / "pilot_v0_1" / "policies.json").read_text(
        encoding="utf-8"
    )
)
sys.path.insert(0, str(UPSTREAM))

from src import engine  # noqa: E402
from src.models import ToolResult  # noqa: E402
from src.utils.persistence import read_json, write_json  # noqa: E402


MODEL_NAME = "Qwen3-1.7B"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "local-experiment"
MAX_ROUNDS = 4
TASK_FILES = POLICIES["tasks"]


def _add_policy_to_initial_context(case_dir: Path, policy: str) -> None:
    instruction = POLICIES["conditions"].get(policy)
    if not instruction:
        return
    round_zero = case_dir / "rounds" / "round-000000"
    for context_path in sorted(round_zero.glob("agent-*/context.json")):
        context = read_json(context_path)
        context["messages"][0]["content"] += (
            "\n\n## Message representation condition\n" + instruction
        )
        write_json(context_path, context)


def _verify_model_shards() -> None:
    model_path = Path(os.environ.get("TLU_MODEL_PATH", ".cache/models/Qwen3-1.7B"))
    for filename, expected_hash in POLICIES["model_shards_sha256"].items():
        path = model_path / filename
        if not path.exists():
            raise SystemExit(f"Missing pinned model shard: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(
                f"Model shard checksum mismatch for {filename}: {actual_hash}"
            )


def _warm_local_model() -> None:
    response = httpx.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 4,
        },
        timeout=180.0,
    )
    response.raise_for_status()


def _disable_communication() -> None:
    original = engine._get_protocol_tools

    def get_tools(protocol: str):
        module = original(protocol)
        execute = module.execute_tool

        def execute_with_channel_intervention(**kwargs: Any):
            if kwargs["tool_call"].tool == "send_message":
                return ToolResult(
                    tool="send_message",
                    parameters=kwargs["tool_call"].parameters,
                    result={"success": False, "message": "channel disabled"},
                    success=False,
                    error="channel disabled",
                )
            return execute(**kwargs)

        return SimpleNamespace(execute_tool=execute_with_channel_intervention)

    engine._get_protocol_tools = get_tools
    return original


def _summarize(case_dir: Path, condition: str, task_name: str, seconds: float) -> dict[str, Any]:
    metadata = read_json(case_dir / "metadata.json")
    results = read_json(case_dir / "results.json")
    message_contents: list[str] = []
    message_file_bytes = 0
    message_count = 0
    for message_path in (case_dir / "rounds").glob("round-*/env/messages/*.json"):
        record = read_json(message_path)
        message_count += 1
        message_contents.append(str(record.get("content", "")))
        message_file_bytes += message_path.stat().st_size
    return {
        "condition": condition,
        "task": task_name,
        "case_dir": str(case_dir.relative_to(ROOT)),
        "success_rate": results["metrics"]["S_success_rate"],
        "partial_correctness": results["metrics"]["P_partial_correctness"],
        "input_tokens": metadata["execution"]["total_input_tokens"],
        "output_tokens": metadata["execution"]["total_output_tokens"],
        "total_tokens": metadata["execution"]["total_input_tokens"]
        + metadata["execution"]["total_output_tokens"],
        "message_count": message_count,
        "message_payload_bytes": sum(len(item.encode("utf-8")) for item in message_contents),
        "simulator_message_file_bytes": message_file_bytes,
        "elapsed_seconds": round(seconds, 3),
        "rounds": metadata["execution"]["current_round"],
        "task_suite_commit": POLICIES["task_suite_commit"],
        "model_revision": POLICIES["model_revision"],
        "submissions": json.dumps(results["submissions"], ensure_ascii=False),
    }


def run_one(task_name: str, condition: str) -> dict[str, Any]:
    intervention = condition == "no_communication"
    previous_get_tools = _disable_communication() if intervention else None
    try:
        case_dir = Path(
            engine.init_case(
                task_file=str(TASKS / task_name),
                protocol="msg",
                model=MODEL_NAME,
                api_base=API_BASE,
                api_key=API_KEY,
                max_rounds=MAX_ROUNDS,
                workspace=str(OUTPUT / "runs"),
            )
        )
        write_json(
            case_dir / "pilot_condition.json",
            {"condition": condition, "task_file": task_name},
        )
        _add_policy_to_initial_context(case_dir, condition)
        started = time.perf_counter()
        for _ in range(MAX_ROUNDS):
            if engine.run_round(str(case_dir)):
                break
        elapsed = time.perf_counter() - started
        return _summarize(case_dir, condition, task_name, elapsed)
    finally:
        if previous_get_tools is not None:
            engine._get_protocol_tools = previous_get_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "unconstrained",
            "concise_nl",
            "autoform_prompt",
            "json_schema",
            "no_communication",
        ],
        choices=[*POLICIES["conditions"], "no_communication"],
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_FILES)
    args = parser.parse_args()
    if not (UPSTREAM / "src" / "engine.py").exists():
        raise SystemExit("Silo-Bench checkout missing; see research/RELATED_WORK.md")
    actual_upstream_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_upstream_commit != POLICIES["task_suite_commit"]:
        raise SystemExit(
            f"Silo-Bench revision mismatch: expected {POLICIES['task_suite_commit']}, "
            f"got {actual_upstream_commit}"
        )
    _verify_model_shards()
    _warm_local_model()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(args.tasks):
        if not (TASKS / task_name).exists():
            raise SystemExit(f"Unknown upstream task file: {task_name}")
        # Rotate the deterministic order so each arm is not always measured first.
        offset = task_index % len(args.conditions)
        task_conditions = args.conditions[offset:] + args.conditions[:offset]
        for condition in task_conditions:
            print(f"Running {task_name} / {condition}", flush=True)
            row = run_one(task_name, condition)
            rows.append(row)
            (OUTPUT / "pilot_runs.jsonl").open("a", encoding="utf-8").write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)

    if rows:
        with (OUTPUT / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Exploratory output: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
