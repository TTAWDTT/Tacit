"""Run v0.13 direct receiver arithmetic controls without simulator tools."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V11 = ROOT / "experiments" / "pilot_v0_11"
sys.path.insert(0, str(V11))

import run_short_shard as base  # noqa: E402
import run_prefixsum as runner  # noqa: E402

PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
CONDITIONS = ("local_prefix", "offset_vector", "combined_receiver")
API_BASE = "http://127.0.0.1:8000/v1"


def prefix(values: list[int]) -> list[int]:
    out, total = [], 0
    for value in values:
        total += value
        out.append(total)
    return out


def prompt_for(condition: str, shard: list[int], offset: int) -> tuple[str, str, list[int]]:
    local = prefix(shard)
    system = "You are an exact integer-arithmetic function. Return only one JSON array of integers. Do not include explanation, Markdown, or XML."
    if condition == "local_prefix":
        user = (
            f"Compute the local inclusive prefix vector p of x={json.dumps(shard)}. "
            "For every i, p[i] is the sum of x[0] through x[i]. Return p as a JSON array."
        )
        expected = local
    elif condition == "offset_vector":
        user = (
            f"The already-computed local prefix vector is p={json.dumps(local)}. "
            f"The exact offset is s={offset}. Return q where q[i]=p[i]+s for every i, as a JSON array."
        )
        expected = [value + offset for value in local]
    elif condition == "combined_receiver":
        user = (
            f"The private local segment is x={json.dumps(shard)} and the exact prior-segment offset is s={offset}. "
            "Compute the local inclusive prefix p, where p[i] is the sum of x[0] through x[i], then return q[i]=p[i]+s as a JSON array."
        )
        expected = [value + offset for value in local]
    else:
        raise ValueError(condition)
    return system, user, expected


def parse_array(text: str) -> list[int] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    return value


def call_model(model: str, system: str, user: str) -> dict[str, Any]:
    response = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {runner.API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 128,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "content": payload["choices"][0]["message"]["content"],
        "input_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
        "output_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Qwen3-4B", "Qwen3-8B"), required=True)
    args = parser.parse_args()
    model = base.configure_model(args.model)
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_13" / args.model
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)

    runner._warm_local_model()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"arithmetic_ladder_runs_{run_id}.jsonl"
    for task_index, task_name in enumerate(runner.TASK_FILES):
        task = runner.read_json(runner.TASKS / task_name)
        shard = task["agent_configs"][1]["input_shard"]
        offset = sum(task["agent_configs"][0]["input_shard"])
        order = list(CONDITIONS)
        if task_index % 2:
            order.reverse()
        for condition in order:
            system, user, expected = prompt_for(condition, shard, offset)
            started = time.perf_counter()
            result = call_model(args.model + "-Q4_K_M", system, user)
            elapsed = time.perf_counter() - started
            predicted = parse_array(result["content"])
            row = {
                "model": model["model"], "condition": condition, "task": task_name,
                "seed": task["metadata"]["seed"], "segment_length": task["metadata"]["segment_length"],
                "shard": shard, "offset": offset, "system_prompt": system, "user_prompt": user,
                "raw_response": result["content"], "parsed_answer": predicted,
                "expected_answer": expected, "parse_success": predicted is not None,
                "exact": predicted == expected, "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"], "elapsed_seconds": round(elapsed, 3),
            }
            with trace_index.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{args.model} {task_name} / {condition}: exact={row['exact']} parse={row['parse_success']}", flush=True)
    print(f"Raw traces: {trace_index}", flush=True)


if __name__ == "__main__":
    main()
