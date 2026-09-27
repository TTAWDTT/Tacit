"""Run the v0.8 role-explicit, lossless PrefixSum protocol comparison.

Requires the pinned local llama.cpp server at 127.0.0.1:8000.
Upstream checkout and generated run data stay in .cache/ (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache" / "upstream-silo-bench"
TASKS = ROOT / "benchmarks" / "prefixsum_v0_2" / "tasks"
OUTPUT = ROOT / ".cache" / "pilot_v0_8"
POLICIES = json.loads(
    (ROOT / "experiments" / "pilot_v0_8" / "policies.json").read_text(
        encoding="utf-8"
    )
)
sys.path.insert(0, str(UPSTREAM))

from src import engine  # noqa: E402
from src.models import ToolCall, ToolResult  # noqa: E402
from src.utils.parsing import _convert_value  # noqa: E402
from src.utils.persistence import read_json, write_json  # noqa: E402


MODEL_NAME = "Qwen3-4B-Q4_K_M"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "local-experiment"
MAX_ROUNDS = POLICIES["max_rounds"]
TASK_FILES = POLICIES["task_files"]


def parse_tool_calls_preserving_message_content(text: str) -> list[ToolCall]:
    """Keep P2P message payloads as literal strings through the XML tool layer.

    The pinned upstream generic parser coerces numeric-only payloads to integers
    and JSON text to Python objects. P2P send_message.content is a wire string,
    so parsing it as a value silently changes the communication protocol.
    """
    tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_name_re = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
    params_block_re = re.compile(r"<parameters>(.*?)</parameters>", re.DOTALL)
    param_re = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
    def keep_json_safe_numbers(value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            if value > (1 << 63) - 1 or value < -(1 << 63):
                return str(value)
            return value
        if isinstance(value, list):
            return [keep_json_safe_numbers(item) for item in value]
        if isinstance(value, dict):
            return {key: keep_json_safe_numbers(item) for key, item in value.items()}
        return value

    calls: list[ToolCall] = []
    for block in tool_call_re.findall(text):
        name_match = tool_name_re.search(block)
        if not name_match:
            continue
        tool_name = name_match.group(1).strip()
        params: dict[str, Any] = {}
        params_match = params_block_re.search(block)
        if params_match:
            for match in param_re.finditer(params_match.group(1)):
                key = match.group(1)
                raw = match.group(2).strip()
                if tool_name == "send_message" and key == "content":
                    params[key] = raw
                else:
                    params[key] = keep_json_safe_numbers(_convert_value(raw))
        calls.append(ToolCall(tool=tool_name, parameters=params))
    return calls


engine.parse_tool_calls = parse_tool_calls_preserving_message_content


def _add_policy_to_initial_context(case_dir: Path, policy: str) -> None:
    if policy == "no_communication":
        return
    instructions = []
    if policy != "unconstrained":
        instructions.append(POLICIES["interaction_scaffold"])
    format_instruction = POLICIES["conditions"].get(policy, "")
    if format_instruction:
        instructions.append("## Message representation condition\n" + format_instruction)
    if not instructions:
        return
    round_zero = case_dir / "rounds" / "round-000000"
    for context_path in sorted(round_zero.glob("agent-*/context.json")):
        context = read_json(context_path)
        context["messages"][0]["content"] += "\n\n" + "\n\n".join(instructions)
        write_json(context_path, context)


def _verify_model_file() -> None:
    path = Path(os.environ.get("TLU_GGUF_PATH", ".cache/models/Qwen3-4B-Q4_K_M.gguf"))
    if not path.exists():
        raise SystemExit(f"Missing pinned GGUF model: {path}")
    if path.stat().st_size != POLICIES["model_file_size_bytes"]:
        raise SystemExit(f"GGUF size mismatch for {path}: {path.stat().st_size}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != POLICIES["model_file_sha256"]:
        raise SystemExit(f"GGUF checksum mismatch: {actual_hash}")


def _verify_task_manifest() -> None:
    manifest_path = TASKS / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != POLICIES["task_manifest_sha256"]:
        raise SystemExit(
            f"Task manifest checksum mismatch: expected {POLICIES['task_manifest_sha256']}, got {digest}"
        )
    manifest = read_json(manifest_path)
    entries = {entry["file"]: entry for entry in manifest["files"]}
    for task_name in TASK_FILES:
        if task_name not in entries:
            raise SystemExit(f"Task absent from pinned manifest: {task_name}")
        task_path = TASKS / task_name
        task_digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
        if task_digest != entries[task_name]["sha256"]:
            raise SystemExit(f"Task checksum mismatch: {task_name}")
        case = read_json(task_path)
        agent0, agent1 = case["agent_configs"]
        expected1 = case["expected_output"]["per_agent_values"][1]
        offset = sum(agent0["input_shard"])
        local_only = []
        running = 0
        for value in agent1["input_shard"]:
            running += value
            local_only.append(running)
        if local_only == expected1 or offset == 0:
            raise SystemExit(f"Agent 1 can solve locally or prior segment is empty: {task_name}")
        if case["metadata"]["split"] != "heldout":
            raise SystemExit(f"Non-held-out task requested in pilot: {task_name}")


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
    task = read_json(TASKS / task_name)
    message_contents: list[str] = []
    message_file_bytes = 0
    message_count = 0
    for message_path in (case_dir / "rounds").glob("round-*/env/messages/*.json"):
        record = read_json(message_path)
        message_count += 1
        message_contents.append(str(record.get("content", "")))
        message_file_bytes += message_path.stat().st_size
    events = []
    for log_path in (case_dir / "logs").glob("agent-*.jsonl"):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    tool_calls = [event for event in events if event.get("event") == "tool_call"]
    sends = [event for event in tool_calls if event.get("tool") == "send_message"]
    agent0_submits = [
        event for event in tool_calls
        if event.get("agent_id") == 0 and event.get("tool") == "submit_result"
    ]
    agent1_submits = [
        event for event in tool_calls
        if event.get("agent_id") == 1 and event.get("tool") == "submit_result"
    ]
    agent0_sends = [event for event in sends if event.get("agent_id") == 0]
    agent1_receives = [
        event for event in tool_calls
        if event.get("agent_id") == 1 and event.get("tool") == "receive_messages"
    ]
    tool_results = [event for event in events if event.get("event") == "tool_result"]
    successful_a0_sends = [
        event for event in tool_results
        if event.get("agent_id") == 0
        and event.get("tool") == "send_message"
        and event.get("result", {}).get("success") is True
    ]
    a1_received_payloads = [
        event for event in tool_results
        if event.get("agent_id") == 1
        and event.get("tool") == "receive_messages"
        and bool(event.get("result", {}).get("messages"))
    ]
    self_sends = [
        event
        for event in tool_calls
        if event.get("tool") == "send_message"
        and event.get("parameters", {}).get("target_id") == event.get("agent_id")
    ]
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
        "send_actions": sum(event.get("tool") == "send_message" for event in tool_calls),
        "agent0_send_to_1_actions": sum(
            event.get("agent_id") == 0
            and event.get("parameters", {}).get("target_id") == 1
            for event in sends
        ),
        "agent1_send_actions": sum(event.get("agent_id") == 1 for event in sends),
        "agent0_successful_sends": len(successful_a0_sends),
        "agent1_received_payloads": sum(
            len(event.get("result", {}).get("messages", [])) for event in a1_received_payloads
        ),
        "agent0_send_before_submit": int(
            bool(agent0_sends)
            and (
                not agent0_submits
                or min(e["timestamp"] for e in agent0_sends)
                < min(e["timestamp"] for e in agent0_submits)
            )
        ),
        "agent1_receive_before_submit": int(
            bool(agent1_receives)
            and bool(agent1_submits)
            and min(e["timestamp"] for e in agent1_receives)
            < min(e["timestamp"] for e in agent1_submits)
        ),
        "agent1_received_payload_before_submit": int(
            bool(a1_received_payloads)
            and bool(agent1_submits)
            and min(e["timestamp"] for e in a1_received_payloads)
            < min(e["timestamp"] for e in agent1_submits)
        ),
        "receive_actions": sum(event.get("tool") == "receive_messages" for event in tool_calls),
        "submit_actions": sum(event.get("tool") == "submit_result" for event in tool_calls),
        "self_send_actions": len(self_sends),
        "message_payload_bytes": sum(len(item.encode("utf-8")) for item in message_contents),
        "simulator_message_file_bytes": message_file_bytes,
        "elapsed_seconds": round(seconds, 3),
        "rounds": metadata["execution"]["current_round"],
        "task_manifest_sha256": POLICIES["task_manifest_sha256"],
        "segment_length": task["metadata"]["segment_length"],
        "task_split": task["metadata"]["split"],
        "upstream_engine_commit": POLICIES["upstream_engine_commit"],
        "model_revision": POLICIES["model_revision"],
        "model_file_sha256": POLICIES["model_file_sha256"],
        "runtime": POLICIES["runtime"],
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
            "scaffold_only",
            "concise_nl",
            "compact_kv",
            "json_schema",
            "binary",
            "full_shard",
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
    if actual_upstream_commit != POLICIES["upstream_engine_commit"]:
        raise SystemExit(
            f"Silo-Bench revision mismatch: expected {POLICIES['upstream_engine_commit']}, "
            f"got {actual_upstream_commit}"
        )
    _verify_task_manifest()
    _verify_model_file()
    _warm_local_model()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(args.tasks):
        if task_name not in TASK_FILES or not (TASKS / task_name).exists():
            raise SystemExit(f"Unknown pinned hidden-sum task file: {task_name}")
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
