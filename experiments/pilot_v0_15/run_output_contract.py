"""Compare JSON-array and XML submit_result answer contracts."""

from __future__ import annotations

import argparse
import json
import re
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
CONDITIONS = ("direct_json_array", "xml_submit_result")
API_BASE = "http://127.0.0.1:8000/v1"


def transcript(offset: int) -> list[dict[str, str]]:
    data = {"messages": [{"from": 0, "content": f"s={offset}", "timestamp": 1}]}
    return [
        {"role": "assistant", "content": "<tool_call><tool>receive_messages</tool><parameters></parameters></tool_call>"},
        {"role": "user", "content": "<tool_result>\n  <tool>receive_messages</tool>\n  <result>" + json.dumps(data) + "</result>\n</tool_result>"},
    ]


def parse_response(text: str, condition: str) -> tuple[list[int] | None, bool]:
    if condition == "direct_json_array":
        candidate = text.strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            return None, False
    else:
        blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if (
            len(blocks) != 1
            or not re.fullmatch(r"\s*<tool_call>.*</tool_call>\s*", text, re.DOTALL)
            or not re.search(r"<tool>\s*submit_result\s*</tool>", blocks[0])
        ):
            return None, False
        answer = re.search(r"<answer>(.*?)</answer>", blocks[0], re.DOTALL)
        if not answer:
            return None, False
        try:
            value = json.loads(answer.group(1).strip())
        except json.JSONDecodeError:
            return None, False
    if not isinstance(value, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in value):
        return None, False
    return value, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Qwen3-8B",), default="Qwen3-8B")
    args = parser.parse_args()
    model = base.configure_model(args.model)
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_15" / args.model
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    runner._warm_local_model()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"output_contract_runs_{run_id}.jsonl"
    for task_index, task_name in enumerate(runner.TASK_FILES):
        task = runner.read_json(runner.TASKS / task_name)
        shard = task["agent_configs"][1]["input_shard"]
        offset = sum(task["agent_configs"][0]["input_shard"])
        expected = task["expected_output"]["per_agent_values"][1]
        common_user = PREREG["common_user_task"].replace("[SHARD]", json.dumps(shard)).replace("OFFSET", str(offset))
        history = [*transcript(offset), {"role": "user", "content": common_user}]
        order = list(CONDITIONS)
        if task_index % 2:
            order.reverse()
        for condition in order:
            instruction = next(item["output_instruction"] for item in PREREG["conditions"] if item["name"] == condition)
            conversation = [*history, {"role": "user", "content": instruction}]
            started = time.perf_counter()
            response = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {runner.API_KEY}"},
                json={
                    "model": "Qwen3-8B-Q4_K_M",
                    "messages": [{"role": "system", "content": PREREG["common_system_prompt"]}, *conversation],
                    "temperature": 0,
                    "max_tokens": 128,
                },
                timeout=180.0,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            elapsed = time.perf_counter() - started
            predicted, valid = parse_response(content, condition)
            row: dict[str, Any] = {
                "model": model["model"], "condition": condition, "task": task_name,
                "seed": task["metadata"]["seed"], "segment_length": task["metadata"]["segment_length"],
                "shard": shard, "offset": offset, "messages": [{"role": "system", "content": PREREG["common_system_prompt"]}, *conversation],
                "raw_response": content, "parsed_answer": predicted, "valid_contract": valid,
                "exact": predicted == expected, "expected_answer": expected,
                "input_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
                "output_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
                "elapsed_seconds": round(elapsed, 3),
            }
            with trace_index.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{task_name} / {condition}: valid={valid} exact={row['exact']}", flush=True)
    print(f"Raw traces: {trace_index}", flush=True)


if __name__ == "__main__":
    main()
