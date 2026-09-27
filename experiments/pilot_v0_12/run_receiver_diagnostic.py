"""Run v0.12's paired ordinary-tool and injected-transcript receiver diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V11 = ROOT / "experiments" / "pilot_v0_11"
sys.path.insert(0, str(V11))

import run_short_shard as base  # noqa: E402
import run_prefixsum as runner  # noqa: E402
from run_oracle_control import oracle_call_llm  # noqa: E402

PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
CONDITIONS = ("ordinary_receive_tool", "injected_successful_receive_transcript")


def inject_successful_receive(case_dir: Path) -> None:
    condition = runner.read_json(case_dir / "pilot_condition.json")
    task_name = condition["task_file"]
    task = runner.read_json(runner.TASKS / task_name)
    subtotal = sum(task["agent_configs"][0]["input_shard"])
    transcript = {
        "messages": [{"from": 0, "content": f"s={subtotal}", "timestamp": 1}]
    }
    receive_call = (
        "<tool_call><tool>receive_messages</tool><parameters>"
        "</parameters></tool_call>"
    )
    tool_result = (
        "<tool_result>\n  <tool>receive_messages</tool>\n  <result>"
        + json.dumps(transcript, ensure_ascii=False)
        + "</result>\n</tool_result>"
    )
    context_path = case_dir / "rounds" / "round-000000" / "agent-001" / "context.json"
    context = runner.read_json(context_path)
    context["messages"].extend([
        {"role": "assistant", "content": receive_call},
        {"role": "user", "content": tool_result},
    ])
    runner.write_json(context_path, context)
    runner.write_json(
        case_dir / "diagnostic_injection.json",
        {
            "condition": "injected_successful_receive_transcript",
            "agent_id": 1,
            "wire_content": f"s={subtotal}",
            "synthetic_context_only": True,
            "counts_as_simulator_receipt": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Qwen3-4B", "Qwen3-8B"), required=True)
    args = parser.parse_args()
    model = base.configure_model(args.model)
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_12" / args.model
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    runner._warm_local_model()

    original_call_llm = runner.engine.call_llm
    original_add_policy = runner._add_policy_to_initial_context
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"receiver_diagnostic_runs_{run_id}.jsonl"

    def add_policy_and_optional_injection(case_dir: Path, policy: str) -> None:
        original_add_policy(case_dir, policy)
        if ACTIVE_CONDITION == "injected_successful_receive_transcript":
            inject_successful_receive(case_dir)

    try:
        runner._add_policy_to_initial_context = add_policy_and_optional_injection
        for task_index, task_name in enumerate(runner.TASK_FILES):
            order = list(CONDITIONS)
            if task_index % 2:
                order.reverse()
            for condition in order:
                global ACTIVE_CONDITION
                ACTIVE_CONDITION = condition
                def oracle_sender_call(*, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
                    initial = next(
                        message["content"] for message in messages
                        if message.get("role") == "user" and "Your private segment" in message.get("content", "")
                    )
                    import re
                    match = re.search(r"You are Agent (\d+) of 2", initial)
                    if match and int(match.group(1)) == 0:
                        return oracle_call_llm(messages=messages, **kwargs)
                    return original_call_llm(messages=messages, **kwargs)

                runner.engine.call_llm = oracle_sender_call
                print(f"Running {args.model} {task_name} / {condition}", flush=True)
                row = runner.run_one(task_name, "compact_kv")
                row.update({"model": model["model"], "diagnostic_condition": condition})
                with trace_index.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        runner.engine.call_llm = original_call_llm
        runner._add_policy_to_initial_context = original_add_policy
    print(f"Raw traces: {trace_index}", flush=True)


ACTIVE_CONDITION = "ordinary_receive_tool"


if __name__ == "__main__":
    main()
