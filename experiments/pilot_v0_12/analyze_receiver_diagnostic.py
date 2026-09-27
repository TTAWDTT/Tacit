"""Analyze v0.12 receiver traces and publish the paired diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_receiver_diagnostic as study
import run_short_shard as base
import run_prefixsum as runner

ROOT = study.ROOT
REPORT_JSON = ROOT / "research" / "PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12.md"
RUN_ROWS = ROOT / "research" / "data" / "PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12_RUNS.jsonl"


def local_prefix(values: list[int]) -> list[int]:
    result, total = [], 0
    for value in values:
        total += value
        result.append(total)
    return result


def extract(row: dict[str, Any]) -> dict[str, Any]:
    task = runner.read_json(runner.TASKS / row["task"])
    shard = task["agent_configs"][1]["input_shard"]
    expected = task["expected_output"]["per_agent_values"][1]
    submissions = {int(item["agent_id"]): item for item in json.loads(row["submissions"])}
    a1 = submissions.get(1, {}).get("answer")
    injected = row["diagnostic_condition"] == "injected_successful_receive_transcript"
    return {
        "agent1_exact": a1 == expected,
        "joint_exact": float(row["success_rate"]) == 1.0,
        "agent1_local_prefix_only": a1 == local_prefix(shard),
        "agent1_received_payload_before_submit": bool(row["agent1_received_payload_before_submit"]),
        "injected_transcript_present": injected,
        "segment_length": int(task["metadata"]["segment_length"]),
        "seed": int(task["metadata"]["seed"]),
        "agent1_answer": a1,
        "agent0_answer": submissions.get(0, {}).get("answer"),
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = len(runner.TASK_FILES) * 2 * len(paths)
    if len(rows) != expected_count:
        raise SystemExit(f"Expected {expected_count} episodes, found {len(rows)}")
    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        record = extract(row)
        grouped[(row["model"], row["diagnostic_condition"], record["segment_length"])].append((row, record))
    summaries = []
    for (model, condition, length), group in sorted(grouped.items()):
        records = [record for _, record in group]
        summaries.append({
            "model": model, "condition": condition, "segment_length": length,
            "episodes": len(group), "receiver_exact": sum(r["agent1_exact"] for r in records),
            "joint_exact": sum(r["joint_exact"] for r in records),
            "received_before_submit": sum(r["agent1_received_payload_before_submit"] for r in records),
            "injected_transcripts": sum(r["injected_transcript_present"] for r in records),
            "local_prefix_only": sum(r["agent1_local_prefix_only"] for r in records),
            "receive_actions": sum(int(row["receive_actions"]) for row, _ in group),
            "mean_backend_tokens": round(sum(int(row["total_tokens"]) for row, _ in group) / len(group), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row, _ in group) / len(group), 2),
        })
    by_task: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        record = extract(row)
        by_task[(row["model"], row["task"], record["segment_length"])][row["diagnostic_condition"]] = record
    paired = []
    for (model, task, length), conditions in by_task.items():
        if set(conditions) != set(study.CONDITIONS):
            raise SystemExit(f"Unpaired diagnostic cell: {model} {task}")
        paired.append({
            "model": model, "task": task, "segment_length": length,
            "ordinary_exact": conditions["ordinary_receive_tool"]["agent1_exact"],
            "injected_exact": conditions["injected_successful_receive_transcript"]["agent1_exact"],
        })
    public_rows = []
    for row in rows:
        record = extract(row)
        injection_path = ROOT / row["case_dir"] / "diagnostic_injection.json"
        injection = json.loads(injection_path.read_text(encoding="utf-8")) if injection_path.exists() else None
        public_rows.append({
            "model": row["model"], "condition": row["diagnostic_condition"], "task": row["task"],
            "seed": record["seed"], "segment_length": record["segment_length"],
            "receiver_exact": record["agent1_exact"], "joint_exact": record["joint_exact"],
            "agent0_answer": record["agent0_answer"], "agent1_answer": record["agent1_answer"],
            "actual_payload_received_before_submit": record["agent1_received_payload_before_submit"],
            "injected_transcript": injection,
            "receive_actions": row["receive_actions"],
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "elapsed_seconds": row["elapsed_seconds"], "rounds": row["rounds"],
        })
    RUN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    RUN_ROWS.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in public_rows), encoding="utf-8", newline="\n")
    return {
        "study": study.PREREG["study"], "task_manifest_sha256": study.PREREG["task_manifest_sha256"],
        "models": study.PREREG["models"], "runtime": study.PREREG["runtime"],
        "summaries_by_model_condition_length": summaries,
        "paired_task_outcomes": paired,
        "raw_trace_files": [str(path.resolve().relative_to(ROOT)).replace("\\", "/") for path in paths],
        "public_run_rows": "research/data/PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12_RUNS.jsonl",
        "limits": study.PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    summaries = result["summaries_by_model_condition_length"]
    lines = [
        "# PrefixSum v0.12 receiver acquisition/application diagnostic", "",
        "A paired diagnostic of ordinary receive-tool use versus an explicitly injected successful receive transcript. The injection is an artificial capability probe; it is neither model-initiated communication nor evidence of protocol superiority.", "",
        f"Task manifest SHA-256: `{result['task_manifest_sha256']}`. Runtime: {result['runtime']}.", "",
        "Each cell has 8 episodes. `Received` counts only payloads actually returned by the simulator before submit; injected transcript rows are counted separately.", "",
        "| Model | Condition | L | Exact receiver outputs | Actual receives | Injected transcript | Local-prefix-only |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        exact_den = item["received_before_submit"] if item["condition"] == "ordinary_receive_tool" else item["episodes"]
        denominator = f"{exact_den}" if item["condition"] == "ordinary_receive_tool" else str(item["episodes"])
        lines.append(f"| {item['model']} | {item['condition']} | {item['segment_length']} | {item['receiver_exact']}/{item['episodes']} | {item['received_before_submit']}/{item['episodes']} | {item['injected_transcripts']}/{item['episodes']} | {item['local_prefix_only']}/{item['episodes']} |")
    lines.extend([
        "", "## Interpretation", "",
        "Compare within each model and length. If injected transcripts improve exact output, the v0.11 failures include an acquisition/tool-action bottleneck; if they do not, receiving the message is insufficient and execution/application remains a bottleneck. Any difference is descriptive for these 24 reused tasks, not a significance claim.",
        "",
        "The injected transcript is a synthetic prior interaction in the harness. Its output must not be counted as successful model tool use, actual message delivery, or a viable communication protocol. Token counts include this extra context and are not an efficiency comparison.",
        "",
        f"Per-episode answers, actual receive status, injection records, and cost traces: [JSONL]({Path('data/PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12_RUNS.jsonl').as_posix()}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    runner.TASKS = ROOT / "benchmarks" / "prefixsum_v0_3" / "tasks"
    manifest = json.loads((runner.TASKS / "manifest.json").read_text(encoding="utf-8"))
    runner.TASK_FILES = [entry["file"] for entry in manifest["files"]]
    result = analyze(args.runs)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(REPORT_MD)


if __name__ == "__main__":
    main()
