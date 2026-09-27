"""Produce a post-hoc diagnostic report from ignored DuoSum pilot logs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".cache" / "pilot_v0_3" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "duosum_v0_1" / "tasks"
MANIFEST = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
MANIFEST_BY_FILE = {entry["file"]: entry for entry in MANIFEST["files"]}
REPORT = ROOT / "research" / "DUOSUM_PILOT_V0_3.md"


def parse_semantic_integer(answer: Any) -> int | None:
    """Parse exact integer answers and exact ``a+b=c`` strings for diagnosis."""
    if isinstance(answer, int) and not isinstance(answer, bool):
        return answer
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    if text.isdecimal():
        return int(text)
    match = re.fullmatch(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)", text)
    if not match:
        return None
    left, right, stated = map(int, match.groups())
    return stated if left + right == stated else None


def load_rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(RAW.open(newline="", encoding="utf-8")))
    for row in rows:
        task = json.loads((TASKS / row["task"]).read_text(encoding="utf-8"))
        gold = task["expected_output"]["per_agent_values"][0]
        submissions = json.loads(row["submissions"])
        row["input_bits"] = int(row["input_bits"])
        row["strict_correct"] = sum(bool(item.get("correct")) for item in submissions)
        row["semantic_correct"] = sum(
            parse_semantic_integer(item.get("answer")) == gold for item in submissions
        )
        row["numeric_submissions"] = sum(
            isinstance(item.get("answer"), int)
            and not isinstance(item.get("answer"), bool)
            for item in submissions
        )
        for field in (
            "total_tokens",
            "message_payload_bytes",
            "message_count",
            "send_actions",
            "receive_actions",
            "submit_actions",
            "self_send_actions",
        ):
            row[field] = int(row[field])
        row["elapsed_seconds"] = float(row["elapsed_seconds"])
    return rows


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def build_report(rows: list[dict[str, Any]]) -> str:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_width_condition: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_width_condition[(row["input_bits"], row["condition"])].append(row)

    conditions = [
        "scaffold_only",
        "concise_nl",
        "compact_kv",
        "json_schema",
        "no_communication",
    ]
    lines = [
        "# DuoSum v0.3 calibration pilot",
        "",
        "**Status:** four calibration episodes, one greedy run per condition and width. This is a feasibility study, not a confirmatory comparison.",
        "",
        "## Pooled outcomes",
        "",
        "Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.",
        "",
        "| Condition | Strict agent success | Semantic exactness | Integer-only submissions | Mean total model tokens | Mean message payload bytes | Mean messages | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        group = by_condition[condition]
        agent_count = 2 * len(group)
        strict = sum(row["strict_correct"] for row in group) / agent_count
        semantic = sum(row["semantic_correct"] for row in group) / agent_count
        numeric = sum(row["numeric_submissions"] for row in group) / agent_count
        lines.append(
            f"| `{condition}` | {strict:.3f} | {semantic:.3f} | {numeric:.3f} | "
            f"{mean(group, 'total_tokens'):.1f} | {mean(group, 'message_payload_bytes'):.1f} | "
            f"{mean(group, 'message_count'):.2f} | {mean(group, 'elapsed_seconds'):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-width results",
            "",
            "Rates are over the two agent submissions in the single episode at that width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.",
            "",
            "| Input width | Condition | Strict success | Semantic exactness | Total model tokens | Payload bytes | Messages |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bits in sorted({row["input_bits"] for row in rows}):
        for condition in conditions:
            group = by_width_condition[(bits, condition)]
            agent_count = 2 * len(group)
            strict = sum(row["strict_correct"] for row in group) / agent_count
            semantic = sum(row["semantic_correct"] for row in group) / agent_count
            lines.append(
                f"| {bits} | `{condition}` | {strict:.2f} | {semantic:.2f} | "
                f"{mean(group, 'total_tokens'):.0f} | {mean(group, 'message_payload_bytes'):.0f} | "
                f"{mean(group, 'message_count'):.1f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The no-communication arm had zero strict and semantic correct submissions across the four calibration episodes. The task construction therefore passed its communication-necessity check for this model and split.",
            "- All JSON-arm submissions were semantically exact under the post-hoc arithmetic parser, while most were rejected by the strict integer tool interface as expression strings. This is a receiver/output-contract failure, not missing information. The parser is diagnostic only; changing the evaluator would require a new benchmark version.",
            "- Compact key-value messages were shortest by UTF-8 payload bytes, but their total model-token cost was not the lowest. Repeated context, format-instruction tokens, extra tool calls, and terminal answer failures mean payload size alone is not an efficiency frontier.",
            "- At the 4-bit calibration episode, JSON and the scaffold-only arm both achieved strict success 1.0; at larger widths, agents generally computed the right arithmetic but emitted expressions instead of an integer. This single seed per width cannot establish a width trend.",
            "- No bandwidth cap was swept, and model-token counts are not information bits. The communication-complexity bound in `docs/THEORY.md` is a wire-bit reference, not a token target.",
            "",
            "## Next revision",
            "",
            "Add a common, format-neutral final-answer contract requiring a bare integer in `submit_result`, then rerun held-out seeds. Keep strict tool success and semantic exactness as separate outcomes. Only after that calibration should message-byte/token budgets be swept and the Pareto frontier compared.",
            "",
            "Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_3/`; this public summary contains aggregate measurements and selected anonymized outcomes only.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    report = build_report(rows)
    REPORT.write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")


if __name__ == "__main__":
    main()
