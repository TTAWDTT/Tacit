"""Analyze the paired v0.14 prompt-scaffold comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_scaffold_context as study

ROOT = study.ROOT
REPORT_JSON = ROOT / "research" / "PREFIXSUM_SCAFFOLD_CONTEXT_V0_14.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_SCAFFOLD_CONTEXT_V0_14.md"
RUN_ROWS = ROOT / "research" / "data" / "PREFIXSUM_SCAFFOLD_CONTEXT_V0_14_RUNS.jsonl"


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = 24 * 2 * len(paths)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} episodes, found {len(rows)}")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["segment_length"]))].append(row)
    summaries = []
    for (condition, length), group in sorted(grouped.items()):
        summaries.append({
            "condition": condition, "segment_length": length, "episodes": len(group),
            "valid_submission": sum(bool(row["valid_submission"]) for row in group),
            "exact": sum(bool(row["exact"]) for row in group),
            "extra_receive_calls": sum("receive_messages" in row["tool_names"] for row in group),
            "mean_input_tokens": round(sum(int(row["input_tokens"]) for row in group) / len(group), 1),
            "mean_output_tokens": round(sum(int(row["output_tokens"]) for row in group) / len(group), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in group) / len(group), 2),
        })
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["task"]][row["condition"]] = row
    transitions = {
        "both_exact": 0, "concise_only": 0, "verbose_only": 0, "neither_exact": 0,
    }
    for task, conditions in paired.items():
        if set(conditions) != set(study.CONDITIONS):
            raise SystemExit(f"Unpaired task: {task}")
        a = bool(conditions["concise_receiver_system"]["exact"])
        b = bool(conditions["verbose_silo_msg_system"]["exact"])
        if a and b:
            transitions["both_exact"] += 1
        elif a:
            transitions["concise_only"] += 1
        elif b:
            transitions["verbose_only"] += 1
        else:
            transitions["neither_exact"] += 1
    RUN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    RUN_ROWS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return {
        "study": study.PREREG["study"], "task_manifest_sha256": study.PREREG["task_manifest_sha256"],
        "model": study.PREREG["model"], "runtime": study.PREREG["runtime"],
        "summaries_by_condition_length": summaries, "paired_transitions": transitions,
        "raw_trace_files": [str(path.resolve().relative_to(ROOT)).replace("\\", "/") for path in paths],
        "public_run_rows": "research/data/PREFIXSUM_SCAFFOLD_CONTEXT_V0_14_RUNS.jsonl",
        "limits": study.PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PrefixSum v0.14 agent-scaffold context diagnostic", "",
        "Paired direct receiver calls with identical data, receive transcript, user task, model, backend, and XML answer contract. The system prompt changes from a concise receiver instruction to the pinned Silo multi-agent tool scaffold.", "",
        f"Task manifest SHA-256: `{result['task_manifest_sha256']}`. Model: {result['model']}. Runtime: {result['runtime']}.", "",
        "Each length-condition cell has 8 cases. Exactness requires one valid XML `submit_result` tool call containing a strict JSON integer array equal to the expected segment.", "",
        "| System condition | L | Valid submit | Exact | Extra receive call | Mean input tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["summaries_by_condition_length"]:
        lines.append(f"| {item['condition']} | {item['segment_length']} | {item['valid_submission']}/{item['episodes']} | {item['exact']}/{item['episodes']} | {item['extra_receive_calls']}/{item['episodes']} | {item['mean_input_tokens']} |")
    tr = result["paired_transitions"]
    lines.extend([
        "", "Paired exact-output outcomes across all 24 tasks:", "",
        f"- Both exact: {tr['both_exact']}/24",
        f"- Concise only: {tr['concise_only']}/24",
        f"- Verbose only: {tr['verbose_only']}/24",
        f"- Neither exact: {tr['neither_exact']}/24",
        "", "## Interpretation", "",
        "This comparison tests compatibility with the agent/tool scaffold, not token length alone. If the concise prompt outperforms, the verbose framework instructions may interfere with this receiver task; if the difference is small, system scaffolding alone is insufficient to explain the simulator/direct-call gap. With 24 reused tasks, treat paired counts descriptively.",
        "",
        "The successful receive transcript is synthetic context and is not counted as model tool use or an actual simulator delivery. No communication format is compared.",
        "",
        f"All prompts, raw outputs, parsed tool calls, expected arrays, and costs: [JSONL]({Path('data/PREFIXSUM_SCAFFOLD_CONTEXT_V0_14_RUNS.jsonl').as_posix()}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.runs)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(REPORT_MD)


if __name__ == "__main__":
    sys.path.insert(0, str(study.V11))
    main()
