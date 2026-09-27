"""Compare concise and pinned Silo tool scaffolds on direct receiver tasks."""

from __future__ import annotations

import argparse
import hashlib
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
from src.utils.prompts import generate_system_prompt  # noqa: E402

PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
CONDITIONS = ("concise_receiver_system", "verbose_silo_msg_system")
API_BASE = "http://127.0.0.1:8000/v1"


def successful_receive_transcript(offset: int) -> list[dict[str, str]]:
    result = {"messages": [{"from": 0, "content": f"s={offset}", "timestamp": 1}]}
    return [
        {
            "role": "assistant",
            "content": "<tool_call><tool>receive_messages</tool><parameters></parameters></tool_call>",
        },
        {
            "role": "user",
            "content": "<tool_result>\n  <tool>receive_messages</tool>\n  <result>"
            + json.dumps(result, ensure_ascii=False)
            + "</result>\n</tool_result>",
        },
    ]


def parse_submission(text: str) -> tuple[list[int] | None, list[str], bool]:
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    names = [
        match.group(1).strip()
        for block in blocks
        if (match := re.search(r"<tool>(.*?)</tool>", block, re.DOTALL))
    ]
    if len(blocks) != 1 or names != ["submit_result"]:
        return None, names, False
    answer_match = re.search(r"<answer>(.*?)</answer>", blocks[0], re.DOTALL)
    if not answer_match:
        return None, names, False
    try:
        answer = json.loads(answer_match.group(1).strip())
    except json.JSONDecodeError:
        return None, names, False
    if not isinstance(answer, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in answer):
        return None, names, False
    return answer, names, True


def call_model(system: str, conversation: list[dict[str, str]]) -> dict[str, Any]:
    response = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {runner.API_KEY}"},
        json={
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "system", "content": system}, *conversation],
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
    parser.add_argument("--model", choices=("Qwen3-8B",), default="Qwen3-8B")
    args = parser.parse_args()
    model = base.configure_model(args.model)
    prompt_source = runner.UPSTREAM / "src" / "utils" / "prompts.py"
    digest = hashlib.sha256(prompt_source.read_bytes()).hexdigest()
    expected_digest = PREREG["conditions"][1]["system_prompt_source_sha256"]
    if digest != expected_digest:
        raise SystemExit(f"Pinned prompt source checksum mismatch: {digest}")
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_14" / args.model
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    runner._warm_local_model()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"scaffold_context_runs_{run_id}.jsonl"
    for task_index, task_name in enumerate(runner.TASK_FILES):
        task = runner.read_json(runner.TASKS / task_name)
        shard = task["agent_configs"][1]["input_shard"]
        offset = sum(task["agent_configs"][0]["input_shard"])
        expected = task["expected_output"]["per_agent_values"][1]
        user = (
            "Compute Agent 1's exact global inclusive prefix segment. Agent 0's segment comes first. "
            f"Your private segment is x={json.dumps(shard)}. The correct Agent 0 subtotal has already "
            f"been received as s={offset}. For every element, compute the local inclusive prefix and add s. "
            "Submit only your own segment in exactly one submit_result XML call, with a JSON array of integers as answer."
        )
        history = [*successful_receive_transcript(offset), {"role": "user", "content": user}]
        order = list(CONDITIONS)
        if task_index % 2:
            order.reverse()
        for condition in order:
            if condition == "concise_receiver_system":
                system = PREREG["conditions"][0]["system_prompt"]
            else:
                system = generate_system_prompt(protocol="msg", agent_id=1, num_agents=2)
                system += "\n\n" + runner.POLICIES["interaction_scaffold"]
                system += "\n\n## Message representation condition\n" + runner.POLICIES["conditions"]["compact_kv"]
            started = time.perf_counter()
            result = call_model(system, history)
            elapsed = time.perf_counter() - started
            answer, tool_names, valid = parse_submission(result["content"])
            row = {
                "model": model["model"], "condition": condition, "task": task_name,
                "seed": task["metadata"]["seed"], "segment_length": task["metadata"]["segment_length"],
                "shard": shard, "offset": offset, "expected_answer": expected,
                "system_prompt": system, "conversation": history,
                "raw_response": result["content"], "parsed_answer": answer,
                "tool_names": tool_names, "valid_submission": valid,
                "exact": answer == expected, "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"], "elapsed_seconds": round(elapsed, 3),
            }
            with trace_index.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{task_name} / {condition}: exact={row['exact']} valid={valid} tools={tool_names}", flush=True)
    print(f"Raw traces: {trace_index}", flush=True)


if __name__ == "__main__":
    main()
