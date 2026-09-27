"""Analyze held-out DuoSum results; keep semantic and tool success separate."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".cache" / "pilot_v0_4" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "duosum_v0_1" / "tasks"
MANIFEST = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
MANIFEST_BY_FILE = {entry["file"]: entry for entry in MANIFEST["files"]}
REPORT = ROOT / "research" / "DUOSUM_PILOT_V0_4.md"


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
        "binary",
        "no_communication",
    ]
    lines = [
        "# DuoSum v0.4 held-out calibration",
        "",
        "**Status:** four held-out episodes, one greedy run per condition and width. This is an exploratory held-out check, not a confirmatory comparison.",
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
            "- The no-communication condition is evaluated on all four episodes; compare its observed strict and semantic rates with each communicating condition rather than assuming the intervention worked.",
            "- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.",
            "- The shared bare-integer instruction did not resolve terminal answer formatting: strict success was 0/8 for scaffold, concise-NL, compact-KV, JSON, and no-communication, and 1/8 for binary. This is an observed interface/model failure, not evidence that the communication messages themselves caused the score gap.",
            "- Semantic exactness was 7/8 for scaffold, 4/8 concise-NL, 5/8 compact-KV, 5/8 JSON, 6/8 binary, and 0/8 no-communication. The one-run-per-cell sample is too small to rank formats; it does show that communication was necessary in these four episodes and that the exact-sum task can reveal information transfer.",
            "- Binary had the shortest mean message payload (6.5 UTF-8 bytes) but did not dominate semantic exactness or total model tokens. This is a descriptive point, not an efficiency frontier: budgets were not matched and setup/context costs are not captured by payload bytes.",
            "- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.",
            "- The four episodes provide only one observation per input width. They are insufficient to estimate a scaling law or stable condition effect.",
            "- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.",
            "",
            "## Next revision",
            "",
            "Before scaling samples or sweeping budgets, separate two effects: (1) message decoding and answer inference, and (2) final answer serialization. Freeze a format-neutral, deterministic answer normalizer/evaluator that accepts an integer or a strictly verified arithmetic expression, report its score alongside Silo's strict score, and preserve the submitted raw answer. Then repeat paired held-out episodes across more seeds and model sizes. Only after reliable task success should equal-byte/equal-token sweeps estimate a communication-efficiency frontier.",
            "",
            "Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_4/`; this public summary contains aggregate measurements and selected anonymized outcomes only.",
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
