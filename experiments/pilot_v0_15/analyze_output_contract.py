"""Analyze paired output-contract results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
REPORT_JSON = ROOT / "research" / "PREFIXSUM_OUTPUT_CONTRACT_V0_15.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_OUTPUT_CONTRACT_V0_15.md"
RUN_ROWS = ROOT / "research" / "data" / "PREFIXSUM_OUTPUT_CONTRACT_V0_15_RUNS.jsonl"
CONDITIONS = ("direct_json_array", "xml_submit_result")


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 48:
        raise SystemExit(f"Expected 48 paired calls, found {len(rows)}")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["condition"], int(row["segment_length"]))].append(row)
        paired[row["task"]][row["condition"]] = row
    summaries = []
    for (condition, length), group in sorted(grouped.items()):
        summaries.append({
            "condition": condition, "segment_length": length, "episodes": len(group),
            "valid_contract": sum(bool(row["valid_contract"]) for row in group),
            "exact": sum(bool(row["exact"]) for row in group),
            "mean_input_tokens": round(sum(row["input_tokens"] for row in group) / len(group), 1),
            "mean_output_tokens": round(sum(row["output_tokens"] for row in group) / len(group), 1),
            "mean_wall_seconds": round(sum(row["elapsed_seconds"] for row in group) / len(group), 2),
        })
    transitions = {"both_exact": 0, "json_only": 0, "xml_only": 0, "neither_exact": 0}
    for task, conditions in paired.items():
        if set(conditions) != set(CONDITIONS):
            raise SystemExit(f"Unpaired task: {task}")
        a = bool(conditions["direct_json_array"]["exact"])
        b = bool(conditions["xml_submit_result"]["exact"])
        if a and b: transitions["both_exact"] += 1
        elif a: transitions["json_only"] += 1
        elif b: transitions["xml_only"] += 1
        else: transitions["neither_exact"] += 1
    RUN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    RUN_ROWS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return {
        "study": PREREG["study"], "task_manifest_sha256": PREREG["task_manifest_sha256"],
        "model": PREREG["model"], "runtime": PREREG["runtime"],
        "summaries_by_condition_length": summaries, "paired_transitions": transitions,
        "raw_trace_files": [str(path.resolve().relative_to(ROOT)).replace("\\", "/") for path in paths],
        "public_run_rows": "research/data/PREFIXSUM_OUTPUT_CONTRACT_V0_15_RUNS.jsonl",
        "limits": PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PrefixSum v0.15 receiver output-contract diagnostic", "",
        "Paired Qwen3-8B direct calls with the same task, inputs, successful receive transcript, and arithmetic prompt. Only the final answer serialization instruction changes between direct JSON and XML `submit_result`.", "",
        f"Task manifest SHA-256: `{result['task_manifest_sha256']}`. Model: {result['model']}. Runtime: {result['runtime']}.", "",
        "Each length-condition cell contains 8 tasks. Contract validity and mathematical exactness are separate outcomes.", "",
        "| Output contract | L | Valid contract | Exact | Mean input tokens | Mean output tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["summaries_by_condition_length"]:
        lines.append(f"| {item['condition']} | {item['segment_length']} | {item['valid_contract']}/{item['episodes']} | {item['exact']}/{item['episodes']} | {item['mean_input_tokens']} | {item['mean_output_tokens']} |")
    transition = result["paired_transitions"]
    lines.extend([
        "", "Paired exact-output transitions over all 24 tasks:", "",
        f"- Both exact: {transition['both_exact']}/24",
        f"- JSON only: {transition['json_only']}/24",
        f"- XML only: {transition['xml_only']}/24",
        f"- Neither exact: {transition['neither_exact']}/24",
        "", "## Interpretation", "",
        "Read output-contract adherence separately from exact task success. A contract can be easier to follow without improving the computation; if JSON produces more exact outputs, the wrapper contributes to the observed gap. Twenty-four reused tasks support only descriptive paired evidence.",
        "",
        "The successful receive transcript is synthetic context, not a model tool call or actual simulator communication. This is a serialization diagnostic, not a message-format or language comparison.",
        "",
        f"Per-episode prompts, answers, raw responses, and costs: [JSONL]({Path('data/PREFIXSUM_OUTPUT_CONTRACT_V0_15_RUNS.jsonl').as_posix()}).",
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
    main()
