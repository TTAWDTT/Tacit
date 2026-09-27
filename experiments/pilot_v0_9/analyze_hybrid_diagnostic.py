"""Analyze completed v0.9 raw traces; makes no model calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import run_hybrid_diagnostic as study

REPORT_MD = study.ROOT / "research" / "PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.md"


def _local_prefix(values: list[int]) -> list[int]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def _episode_details(row: dict[str, Any]) -> dict[str, Any]:
    task = study.runner.read_json(study.runner.TASKS / row["task"])
    shards = [agent["input_shard"] for agent in task["agent_configs"]]
    gold = task["expected_output"]["per_agent_values"]
    submissions = {int(item["agent_id"]): item for item in json.loads(row["submissions"])}
    messages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((study.ROOT / row["case_dir"]).glob("rounds/round-*/env/messages/*.json"))
    ]
    message = messages[0] if messages else None
    match = re.fullmatch(r"s=(\d+)", str(message["content"])) if message else None
    wire_value = int(match.group(1)) if match else None
    true_subtotal = sum(shards[0])
    a1_answer = submissions.get(1, {}).get("answer")
    a0_answer = submissions.get(0, {}).get("answer")
    local = _local_prefix(shards[1])
    agent1_exact = bool(submissions.get(1, {}).get("correct"))
    local_only = a1_answer == local
    receiver_used_wire = (
        wire_value is not None
        and a1_answer == [wire_value + value for value in local]
    )
    return {
        "agent0_exact": bool(submissions.get(0, {}).get("correct")),
        "agent1_exact": agent1_exact,
        "agent1_local_prefix_only": local_only,
        "agent1_answer_matches_wire_offset": receiver_used_wire,
        "message_syntax": match is not None,
        "message_value_faithful": wire_value == true_subtotal,
        "oracle_receiver_calculation_exact": (
            receiver_used_wire if row["variant"] == "qwen_sender_oracle_receiver" else None
        ),
        "agent1_received_before_submit": bool(row["agent1_received_payload_before_submit"]),
        "fully_correct_episode": float(row["success_rate"]) == 1.0,
        "segment_length": int(task["metadata"]["segment_length"]),
        "message_value": wire_value,
        "true_subtotal": true_subtotal,
        "a0_answer": a0_answer,
        "a1_answer": a1_answer,
    }


def analyze(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = 2 * len(study.runner.TASK_FILES)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} complete cells, found {len(rows)}")
    variants = []
    for name in study.VARIANTS:
        group = [row for row in rows if row["variant"] == name]
        if len(group) != len(study.runner.TASK_FILES):
            raise SystemExit(f"Expected {len(study.runner.TASK_FILES)} rows for {name}, found {len(group)}")
        details = [_episode_details(row) for row in group]
        received = [detail for detail in details if detail["agent1_received_before_submit"]]
        variants.append(
            {
                "variant": name,
                "episodes": len(group),
                "agent0_exact": sum(detail["agent0_exact"] for detail in details),
                "agent1_exact": sum(detail["agent1_exact"] for detail in details),
                "fully_correct_episodes": sum(detail["fully_correct_episode"] for detail in details),
                "delivered_messages": sum(int(row["message_count"]) for row in group),
                "agent1_received_before_submit": len(received),
                "agent1_exact_given_received": sum(detail["agent1_exact"] for detail in received),
                "agent1_local_prefix_only": sum(detail["agent1_local_prefix_only"] for detail in details),
                "agent1_matches_wire_offset": sum(detail["agent1_answer_matches_wire_offset"] for detail in details),
                "compact_kv_syntax": sum(detail["message_syntax"] for detail in details),
                "subtotal_value_fidelity": sum(detail["message_value_faithful"] for detail in details),
                "oracle_receiver_applied_wire_value": sum(
                    detail["oracle_receiver_calculation_exact"] is True for detail in details
                ) if name == "qwen_sender_oracle_receiver" else None,
                "mean_backend_tokens": round(sum(int(row["total_tokens"]) for row in group) / len(group), 1),
                "mean_payload_bytes": round(sum(int(row["message_payload_bytes"]) for row in group) / len(group), 1),
                "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in group) / len(group), 2),
                "agent1_output_categories": dict(Counter(
                    "no_submission" if detail["a1_answer"] is None
                    else "exact_local_prefix_only" if detail["agent1_local_prefix_only"]
                    else "exact_global_prefix" if detail["agent1_exact"]
                    else "other_incorrect_array"
                    for detail in details
                )),
            }
        )
    return {
        "study": "paired_hybrid_role_diagnostic_v0_9",
        "model_revision": study.runner.POLICIES["model_revision"],
        "model_file_sha256": study.runner.POLICIES["model_file_sha256"],
        "task_manifest_sha256": study.runner.POLICIES["task_manifest_sha256"],
        "upstream_engine_commit": study.runner.POLICIES["upstream_engine_commit"],
        "runtime": study.runner.POLICIES["runtime"],
        "condition": "compact_kv",
        "variants": variants,
        "limits": json.loads((study.HERE / "preregistration.json").read_text(encoding="utf-8"))["limits"],
        "raw_trace_file": f".cache/pilot_v0_9/{path.name}",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    by_name = {item["variant"]: item for item in summary["variants"]}
    receiver = by_name["oracle_sender_qwen_receiver"]
    sender = by_name["qwen_sender_oracle_receiver"]
    receiver_categories = receiver["agent1_output_categories"]
    return "\n".join(
        [
            "# PrefixSum v0.9 hybrid role diagnostic",
            "",
            "**Design:** paired diagnostic on the 12 v0.8 task seeds, using one greedy Qwen3-4B Q4_K_M run per task and hybrid arm. The message condition remains compact-KV (`s=<sum>`). The preregistration was committed before these model calls. This is not new held-out evidence or a language ranking.",
            "",
            f"Pinned model checksum: `{summary['model_file_sha256']}`. Task manifest checksum: `{summary['task_manifest_sha256']}`. Runtime: {summary['runtime']}.",
            "",
            "## Results",
            "",
            "| Hybrid arm | Agent 0 exact | Agent 1 exact | Joint exact episodes | A1 received before submit | Compact-KV syntax | Correct subtotal | A1 exactly followed wire offset | Mean backend tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| Oracle sender + Qwen receiver | {receiver['agent0_exact']}/{receiver['episodes']} | {receiver['agent1_exact']}/{receiver['episodes']} | {receiver['fully_correct_episodes']}/{receiver['episodes']} | {receiver['agent1_received_before_submit']}/{receiver['episodes']} | {receiver['compact_kv_syntax']}/{receiver['delivered_messages']} | {receiver['subtotal_value_fidelity']}/{receiver['delivered_messages']} | {receiver['agent1_matches_wire_offset']}/{receiver['episodes']} | {receiver['mean_backend_tokens']:.0f} |",
            f"| Qwen sender + oracle receiver | {sender['agent0_exact']}/{sender['episodes']} | {sender['agent1_exact']}/{sender['episodes']} | {sender['fully_correct_episodes']}/{sender['episodes']} | {sender['agent1_received_before_submit']}/{sender['episodes']} | {sender['compact_kv_syntax']}/{sender['delivered_messages']} | {sender['subtotal_value_fidelity']}/{sender['delivered_messages']} | {sender['agent1_matches_wire_offset']}/{sender['episodes']} | {sender['mean_backend_tokens']:.0f} |",
            "",
            "The oracle sender supplied a correct subtotal in every episode and submitted Agent 0's exact segment in 12/12. Qwen Agent 1 received the non-empty payload before submitting in 11/12, but returned the exact global segment in 0/11 of those received-message cases. Its outputs were exact local-only prefix sums in 4/12, other incorrect arrays in 7/12, and missing in 1/12. This isolates a receiver-side failure after correct information was available; it does not prove the model never parsed any message token.",
            "",
            "With Qwen as sender, all 12 messages had compact-KV syntax and were delivered, but none contained Agent 0's true subtotal. Agent 0's local prefix array was exact in 3/12. The deterministic receiver's Agent 1 outputs exactly matched the received wire subtotal plus local prefixes in 12/12, confirming the receiver control used the sent value; the joint task still failed in all 12 because the sender information and/or Agent 0 output was wrong.",
            "",
            "## Interpretation and limits",
            "",
            "The paired controls support two independent failure modes in this setup: Qwen3-4B often computes or reports the wrong subtotal as sender, and it does not reliably apply even a correct subtotal as receiver. The sender and receiver bottlenecks were not separable from v0.8 alone; this hybrid makes their presence visible. The benchmark therefore remains a model/tool-following diagnostic, not evidence for or against any general communication language.",
            "",
            "This reuses the same 12 cases as v0.8, has one greedy run per arm, and changes which role invokes the model. Backend-token means are descriptive only: the model handles different role prompts and histories, and the oracle role has no model token cost. No efficiency comparison is valid. See the frozen [preregistration](../experiments/pilot_v0_9/preregistration.json), [runner](../experiments/pilot_v0_9/run_hybrid_diagnostic.py), and [analyzer](../experiments/pilot_v0_9/analyze_hybrid_diagnostic.py).",
            "",
            f"Raw local traces are under ignored `{summary['raw_trace_file']}`. Aggregate data are also preserved in [JSON](PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.json).",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, help="specific hybrid_runs_*.jsonl file")
    args = parser.parse_args()
    path = args.runs or max((study.ROOT / ".cache" / "pilot_v0_9").glob("hybrid_runs_*.jsonl"), key=lambda p: p.stat().st_mtime)
    summary = analyze(path)
    study.REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
