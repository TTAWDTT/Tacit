"""Run a paired Qwen/oracle role diagnostic on the frozen PrefixSum cases."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V08 = ROOT / "experiments" / "pilot_v0_8"
sys.path.insert(0, str(V08))

import run_prefixsum as runner  # noqa: E402
from analyze_prefixsum import message_adherence  # noqa: E402
from run_oracle_control import oracle_call_llm  # noqa: E402


VARIANTS = {
    "oracle_sender_qwen_receiver": 0,
    "qwen_sender_oracle_receiver": 1,
}
REPORT = ROOT / "research" / "PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.json"
PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))


def _agent_id(messages: list[dict[str, Any]]) -> int:
    initial = next(
        message["content"]
        for message in messages
        if message.get("role") == "user" and "Your private segment" in message.get("content", "")
    )
    import re

    match = re.search(r"You are Agent (\d+) of 2", initial)
    if not match:
        raise ValueError("Unable to identify agent in model request")
    return int(match.group(1))


def _summarize_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    strict = {0: 0, 1: 0}
    messages = syntax = faithful = 0
    agent1_received = agent1_received_correct = 0
    for row in rows:
        task = runner.read_json(runner.TASKS / row["task"])
        submissions = json.loads(row["submissions"])
        by_agent = {int(item["agent_id"]): item for item in submissions}
        for agent_id in (0, 1):
            strict[agent_id] += int(bool(by_agent.get(agent_id, {}).get("correct")))
        message_syntax, message_faithful, message_total = message_adherence(row, task)
        syntax += message_syntax
        faithful += message_faithful
        messages += message_total
        if int(row["agent1_received_payload_before_submit"]):
            agent1_received += 1
            agent1_received_correct += int(bool(by_agent.get(1, {}).get("correct")))

    return {
        "variant": variant,
        "episodes": len(rows),
        "agent0_exact": f"{strict[0]}/{len(rows)}",
        "agent1_exact": f"{strict[1]}/{len(rows)}",
        "fully_correct_episodes": f"{sum(float(row['success_rate']) == 1.0 for row in rows)}/{len(rows)}",
        "delivered_messages": sum(int(row["message_count"]) for row in rows),
        "agent1_received_before_submit": f"{agent1_received}/{len(rows)}",
        "agent1_exact_given_received": f"{agent1_received_correct}/{agent1_received}" if agent1_received else "n/a",
        "compact_kv_syntax": f"{syntax}/{messages}" if messages else "n/a",
        "subtotal_value_fidelity": f"{faithful}/{messages}" if messages else "n/a",
        "mean_backend_tokens": sum(int(row["total_tokens"]) for row in rows) / len(rows),
        "mean_payload_bytes": sum(int(row["message_payload_bytes"]) for row in rows) / len(rows),
        "mean_wall_seconds": sum(float(row["elapsed_seconds"]) for row in rows) / len(rows),
    }


def main() -> None:
    if PREREG["task_manifest_sha256"] != runner.POLICIES["task_manifest_sha256"]:
        raise SystemExit("Preregistered task manifest does not match the pinned runner")
    if PREREG["message_condition"] != "compact_kv" or set(VARIANTS) != {
        "oracle_sender_qwen_receiver", "qwen_sender_oracle_receiver"
    }:
        raise SystemExit("Runner conditions differ from the preregistration")
    if PREREG["model_file_sha256"] != runner.POLICIES["model_file_sha256"]:
        raise SystemExit("Preregistered model checksum does not match the pinned runner")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(runner.UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_commit != runner.POLICIES["upstream_engine_commit"]:
        raise SystemExit(f"Pinned upstream mismatch: {upstream_commit}")
    runner._verify_task_manifest()
    runner._verify_model_file()
    runner._warm_local_model()

    original_call_llm = runner.engine.call_llm
    original_output = runner.OUTPUT
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_9"
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {key: [] for key in VARIANTS}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"hybrid_runs_{run_id}.jsonl"
    try:
        for task_index, task_name in enumerate(runner.TASK_FILES):
            variant_order = list(VARIANTS)
            if task_index % 2:
                variant_order.reverse()
            for variant in variant_order:
                oracle_agent_id = VARIANTS[variant]

                def hybrid_call_llm(*, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
                    if _agent_id(messages) == oracle_agent_id:
                        return oracle_call_llm(messages=messages, **kwargs)
                    return original_call_llm(messages=messages, **kwargs)

                runner.engine.call_llm = hybrid_call_llm
                print(f"Running {task_name} / {variant}", flush=True)
                row = runner.run_one(task_name, "compact_kv")
                row["variant"] = variant
                results[variant].append(row)
                with trace_index.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        runner.engine.call_llm = original_call_llm
        runner.OUTPUT = original_output

    summary = {
        "study": "paired_hybrid_role_diagnostic_v0_9",
        "model_revision": runner.POLICIES["model_revision"],
        "model_file_sha256": runner.POLICIES["model_file_sha256"],
        "task_manifest_sha256": runner.POLICIES["task_manifest_sha256"],
        "upstream_engine_commit": runner.POLICIES["upstream_engine_commit"],
        "runtime": runner.POLICIES["runtime"],
        "condition": "compact_kv",
        "variants": [
            _summarize_variant(results[name], name)
            for name in VARIANTS
        ],
        "limits": [
            "same 12 tasks as v0.8; paired diagnosis, not new held-out evidence",
            "one greedy Qwen run per task and role variant",
            "oracle agents are deterministic controls, not model results",
            "role prompts, model call counts, and conversation lengths differ; no efficiency ranking",
        ],
    }
    REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
