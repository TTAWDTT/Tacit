"""Run direct arithmetic controls and a Silo-engine receiver episode for Qwen3-14B."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V08 = ROOT / "experiments" / "pilot_v0_8"
sys.path.insert(0, str(V08))
import run_prefixsum as runner  # noqa: E402
from run_oracle_control import oracle_call_llm  # noqa: E402

PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
MODEL_NAME = "Qwen3-14B-Q4_K_M"
API_BASE = "http://127.0.0.1:8000/v1"
DIRECT_CONDITIONS = ("local_prefix", "offset_vector", "combined_receiver")


def prefix(values: list[int]) -> list[int]:
    result, total = [], 0
    for value in values:
        total += value
        result.append(total)
    return result


def direct_prompt(condition: str, shard: list[int], offset: int) -> tuple[str, str, list[int]]:
    local = prefix(shard)
    system = "You are an exact integer-arithmetic function. Return only one JSON array of integers. Do not include explanation, Markdown, or XML."
    if condition == "local_prefix":
        user = f"Compute the local inclusive prefix vector p of x={json.dumps(shard)}. For every i, p[i] is the sum of x[0] through x[i]. Return p as a JSON array."
        expected = local
    elif condition == "offset_vector":
        user = f"The already-computed local prefix vector is p={json.dumps(local)}. The exact offset is s={offset}. Return q where q[i]=p[i]+s for every element, as a JSON array."
        expected = [value + offset for value in local]
    elif condition == "combined_receiver":
        user = f"The private local segment is x={json.dumps(shard)} and the exact prior-segment offset is s={offset}. Compute the local inclusive prefix p, where p[i] is the sum of x[0] through x[i], then return q[i]=p[i]+s as a JSON array."
        expected = [value + offset for value in local]
    else:
        raise ValueError(condition)
    return system, user, expected


def parse_array(text: str) -> list[int] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in value):
        return None
    return value


def configure() -> None:
    runner.TASKS = ROOT / "benchmarks" / "prefixsum_v0_3" / "tasks"
    manifest = json.loads((runner.TASKS / "manifest.json").read_text(encoding="utf-8"))
    runner.TASK_FILES = [entry["file"] for entry in manifest["files"]]
    runner.POLICIES["task_files"] = runner.TASK_FILES
    runner.POLICIES["task_manifest_sha256"] = PREREG["task_manifest_sha256"]
    runner.POLICIES["model_file_sha256"] = PREREG["model_file_sha256"]
    runner.POLICIES["model_file_size_bytes"] = PREREG["model_file_size_bytes"]
    runner.POLICIES["model_revision"] = PREREG["model_repository_revision"]
    runner.POLICIES["runtime"] = PREREG["runtime"]
    runner.MODEL_NAME = MODEL_NAME
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_16" / "hybrid"
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    runner._verify_task_manifest()
    runner._verify_model_file()
    expected = "e74127782ed1c42fff474249961f022c063d76f2"
    actual = subprocess.check_output(["git", "-C", str(runner.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise SystemExit(f"Pinned Silo-Bench engine mismatch: {actual}")


def run_direct(trace: Path) -> None:
    for task_index, task_name in enumerate(runner.TASK_FILES):
        task = runner.read_json(runner.TASKS / task_name)
        shard = task["agent_configs"][1]["input_shard"]
        offset = sum(task["agent_configs"][0]["input_shard"])
        conditions = list(DIRECT_CONDITIONS)
        if task_index % 2:
            conditions.reverse()
        for condition in conditions:
            system, user, expected = direct_prompt(condition, shard, offset)
            start = time.perf_counter()
            response = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {runner.API_KEY}"},
                json={"model": MODEL_NAME, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": 128},
                timeout=300.0,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            predicted = parse_array(content)
            row = {
                "component": "direct", "model": PREREG["model"], "condition": condition,
                "task": task_name, "seed": task["metadata"]["seed"], "segment_length": task["metadata"]["segment_length"],
                "shard": shard, "offset": offset, "system_prompt": system, "user_prompt": user,
                "raw_response": content, "parsed_answer": predicted, "expected_answer": expected,
                "parse_success": predicted is not None, "exact": predicted == expected,
                "input_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
                "output_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
            with trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"direct {task_name}/{condition}: exact={row['exact']}", flush=True)


def run_hybrid(trace: Path) -> None:
    original = runner.engine.call_llm

    def oracle_sender_hybrid(*, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        initial = next(message["content"] for message in messages if message.get("role") == "user" and "Your private segment" in message.get("content", ""))
        match = re.search(r"You are Agent (\d+) of 2", initial)
        if match and int(match.group(1)) == 0:
            return oracle_call_llm(messages=messages, **kwargs)
        return original(messages=messages, **kwargs)

    try:
        runner.engine.call_llm = oracle_sender_hybrid
        for task_name in runner.TASK_FILES:
            row = runner.run_one(task_name, "compact_kv")
            row.update({"component": "simulator_receiver", "model": PREREG["model"], "variant": "oracle_sender_qwen_receiver_14b"})
            with trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"hybrid {task_name}: success={row['success_rate']} received={row['agent1_received_payload_before_submit']}", flush=True)
    finally:
        runner.engine.call_llm = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("direct", "hybrid", "all"), default="all")
    args = parser.parse_args()
    configure()
    runner._warm_local_model()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    direct_trace = ROOT / ".cache" / "pilot_v0_16" / "direct" / f"direct_runs_{stamp}.jsonl"
    hybrid_trace = ROOT / ".cache" / "pilot_v0_16" / "hybrid" / f"hybrid_runs_{stamp}.jsonl"
    direct_trace.parent.mkdir(parents=True, exist_ok=True)
    hybrid_trace.parent.mkdir(parents=True, exist_ok=True)
    if args.component in ("direct", "all"):
        run_direct(direct_trace)
    if args.component in ("hybrid", "all"):
        run_hybrid(hybrid_trace)
    print(json.dumps({"direct_trace": str(direct_trace), "hybrid_trace": str(hybrid_trace)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
