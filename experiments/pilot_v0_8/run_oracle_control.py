"""Drive the pinned PrefixSum message engine with deterministic oracle agents.

This control tests task/scorer/message-tool mechanics without any model calls.
It is not an LLM result and is excluded from protocol comparisons.
"""

from __future__ import annotations

import json
import re
from typing import Any

import run_prefixsum as runner


def _latest_receive_result(messages: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    for message in reversed(messages):
        content = message.get("content", "")
        if "<tool>receive_messages</tool>" not in content:
            continue
        match = re.search(r"<result>(.*?)</result>", content, re.DOTALL)
        if not match:
            continue
        try:
            result = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        received = result.get("messages")
        if isinstance(received, list):
            return received
    return None


def oracle_call_llm(**kwargs: Any) -> dict[str, Any]:
    messages = kwargs["messages"]
    initial = next(
        message["content"]
        for message in messages
        if message.get("role") == "user" and "Your private segment" in message.get("content", "")
    )
    agent_match = re.search(r"You are Agent (\d+) of 2", initial)
    shard_match = re.search(r"Your private segment .*?: (\[[^\]]*\])", initial)
    if not agent_match or not shard_match:
        raise ValueError("Oracle could not recover its pinned agent identity or private shard")
    agent_id = int(agent_match.group(1))
    shard = json.loads(shard_match.group(1))

    if agent_id == 0:
        subtotal = sum(shard)
        prefix: list[int] = []
        running = 0
        for value in shard:
            running += value
            prefix.append(running)
        content = (
            "<tool_call><tool>send_message</tool><parameters>"
            f"<target_id>1</target_id><content>s={subtotal}</content>"
            "</parameters></tool_call>"
            "<tool_call><tool>submit_result</tool><parameters>"
            f"<answer>{json.dumps(prefix)}</answer>"
            "</parameters></tool_call>"
        )
    else:
        received = _latest_receive_result(messages)
        if not received:
            waited = any(
                message.get("role") == "assistant" and "<tool>wait</tool>" in message.get("content", "")
                for message in messages
            )
            tool = "receive_messages" if waited else (
                "wait" if received == [] else "receive_messages"
            )
            content = f"<tool_call><tool>{tool}</tool><parameters></parameters></tool_call>"
        else:
            offset_match = re.fullmatch(r"s=(\d+)", str(received[0].get("content", "")))
            if not offset_match or int(received[0].get("from", -1)) != 0:
                prefix = []
            else:
                offset = int(offset_match.group(1))
                prefix = []
                running = offset
                for value in shard:
                    running += value
                    prefix.append(running)
            content = (
                "<tool_call><tool>submit_result</tool><parameters>"
                f"<answer>{json.dumps(prefix)}</answer>"
                "</parameters></tool_call>"
            )

    return {"content": content, "input_tokens": 0, "output_tokens": 0}


def main() -> None:
    runner._verify_task_manifest()
    original_call_llm = runner.engine.call_llm
    runner.engine.call_llm = oracle_call_llm
    try:
        rows = [runner.run_one(task_name, "compact_kv") for task_name in runner.TASK_FILES]
    finally:
        runner.engine.call_llm = original_call_llm

    successful = sum(row["success_rate"] == 1.0 for row in rows)
    delivered = sum(row["message_count"] for row in rows)
    faithful = 0
    for row in rows:
        task = runner.read_json(runner.TASKS / row["task"])
        message_files = list((runner.ROOT / row["case_dir"]).glob("rounds/round-*/env/messages/*.json"))
        if len(message_files) != 1:
            raise AssertionError(f"Expected one oracle message in {row['task']}, got {len(message_files)}")
        message = runner.read_json(message_files[0])
        expected = sum(task["agent_configs"][0]["input_shard"])
        faithful += int(message["content"] == f"s={expected}" and message["sender_id"] == 0 and message["recipient_id"] == 1)

    summary = {
        "control": "deterministic_oracle_no_model_calls",
        "episodes": len(rows),
        "fully_correct_episodes": successful,
        "delivered_messages": delivered,
        "faithful_subtotal_messages": faithful,
        "mean_rounds": sum(row["rounds"] for row in rows) / len(rows),
        "model_backend_input_tokens": sum(row["input_tokens"] for row in rows),
        "model_backend_output_tokens": sum(row["output_tokens"] for row in rows),
        "task_manifest_sha256": runner.POLICIES["task_manifest_sha256"],
        "upstream_engine_commit": runner.POLICIES["upstream_engine_commit"],
    }
    record_path = runner.ROOT / "research" / "PREFIXSUM_ORACLE_CONTROL_V0_1.json"
    record_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    if successful != len(rows) or delivered != len(rows) or faithful != len(rows):
        raise SystemExit("Oracle control failed: inspect generated traces and scorer output")


if __name__ == "__main__":
    main()
