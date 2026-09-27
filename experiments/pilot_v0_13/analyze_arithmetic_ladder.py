"""Analyze direct arithmetic ladder traces and publish per-episode outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_arithmetic_ladder as study

ROOT = study.ROOT
REPORT_JSON = ROOT / "research" / "PREFIXSUM_ARITHMETIC_LADDER_V0_13.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_ARITHMETIC_LADDER_V0_13.md"
RUN_ROWS = ROOT / "research" / "data" / "PREFIXSUM_ARITHMETIC_LADDER_V0_13_RUNS.jsonl"


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_total = len(study.base.runner.TASK_FILES) * len(study.CONDITIONS) * len(paths)
    if len(rows) != expected_total:
        raise SystemExit(f"Expected {expected_total} episodes, found {len(rows)}")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"], int(row["segment_length"]))].append(row)
    summaries = []
    for (model, condition, length), group in sorted(grouped.items()):
        summaries.append({
            "model": model, "condition": condition, "segment_length": length,
            "episodes": len(group), "parse_success": sum(bool(row["parse_success"]) for row in group),
            "exact": sum(bool(row["exact"]) for row in group),
            "mean_input_tokens": round(sum(int(row["input_tokens"]) for row in group) / len(group), 1),
            "mean_output_tokens": round(sum(int(row["output_tokens"]) for row in group) / len(group), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in group) / len(group), 2),
        })
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["model"], row["task"])][row["condition"]] = row
    transitions = []
    for model in sorted({key[0] for key in paired}):
        tasks = [conditions for (task_model, _), conditions in paired.items() if task_model == model]
        local = sum(bool(task["local_prefix"]["exact"]) for task in tasks)
        offset = sum(bool(task["offset_vector"]["exact"]) for task in tasks)
        combined = sum(bool(task["combined_receiver"]["exact"]) for task in tasks)
        transitions.append({"model": model, "tasks": len(tasks), "local_prefix_exact": local, "offset_vector_exact": offset, "combined_receiver_exact": combined})
    RUN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    RUN_ROWS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return {
        "study": study.PREREG["study"], "task_manifest_sha256": study.PREREG["task_manifest_sha256"],
        "models": study.PREREG["models"], "runtime": study.PREREG["runtime"],
        "summaries_by_model_condition_length": summaries,
        "paired_task_totals_by_model": transitions,
        "raw_trace_files": [str(path.resolve().relative_to(ROOT)).replace("\\", "/") for path in paths],
        "public_run_rows": "research/data/PREFIXSUM_ARITHMETIC_LADDER_V0_13_RUNS.jsonl",
        "limits": study.PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PrefixSum v0.13 receiver arithmetic ladder", "",
        "Direct single-agent arithmetic controls, without tools or a multi-agent simulator. The same short-shard inputs are reused to isolate local prefix computation, scalar-offset addition to a supplied vector, and their composition.", "",
        f"Task manifest SHA-256: `{result['task_manifest_sha256']}`. Runtime: {result['runtime']}.", "",
        "Each cell has 8 cases. A response counts as parseable only if the full response is a JSON list of non-boolean integers.", "",
        "| Model | Operation | L | Parseable | Exact | Mean input tokens | Mean output tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["summaries_by_model_condition_length"]:
        lines.append(f"| {item['model']} | {item['condition']} | {item['segment_length']} | {item['parse_success']}/{item['episodes']} | {item['exact']}/{item['episodes']} | {item['mean_input_tokens']} | {item['mean_output_tokens']} |")
    lines.extend([
        "", "## Paired totals across all 24 cases", "",
        "| Model | Local prefix exact | Offset-vector exact | Combined receiver exact |",
        "|---|---:|---:|---:|",
    ])
    for item in result["paired_task_totals_by_model"]:
        lines.append(f"| {item['model']} | {item['local_prefix_exact']}/24 | {item['offset_vector_exact']}/24 | {item['combined_receiver_exact']}/24 |")
    lines.extend([
        "", "## Interpretation", "",
        "Use differences between these arithmetic controls to locate the failing operation. High local-prefix and offset-vector accuracy with low combined accuracy would implicate composition; low offset-vector accuracy despite supplied prefixes would show that scalar application itself remains difficult. High direct accuracy would point back toward multi-agent interaction or harness context. These controls cannot establish message-format efficiency or generalization beyond this task family.",
        "",
        "This is a narrow diagnostic on reused task inputs with one greedy output each. Models and checkpoints are not a causal size comparison.",
        "",
        f"All prompts, raw responses, strict parses, expected arrays, and cost fields: [JSONL]({Path('data/PREFIXSUM_ARITHMETIC_LADDER_V0_13_RUNS.jsonl').as_posix()}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(study.V11))
    import run_prefixsum as runner
    runner.TASKS = ROOT / "benchmarks" / "prefixsum_v0_3" / "tasks"
    manifest = json.loads((runner.TASKS / "manifest.json").read_text(encoding="utf-8"))
    runner.TASK_FILES = [entry["file"] for entry in manifest["files"]]
    result = analyze(args.runs)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(REPORT_MD)


if __name__ == "__main__":
    main()
