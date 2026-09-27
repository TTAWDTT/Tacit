"""Analyze v0.5 DuoSum outcomes, serialization, and fixed-format fidelity."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.duosum_v0_1.grading import answer_is_correct, decode_message


RAW = ROOT / ".cache" / "pilot_v0_5" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "duosum_v0_1" / "tasks"
MANIFEST = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
MANIFEST_BY_FILE = {entry["file"]: entry for entry in MANIFEST["files"]}
POLICIES = json.loads((ROOT / "experiments" / "pilot_v0_5" / "policies.json").read_text(encoding="utf-8"))
REPORT = ROOT / "research" / "DUOSUM_PILOT_V0_5.md"


def message_format_adherence(row: dict[str, Any], task: dict[str, Any]) -> tuple[int, int, int]:
    """Count syntactically valid messages and decoded sender-value matches."""
    condition = row["condition"]
    if condition in {"scaffold_only", "autoform", "no_communication"}:
        return (0, 0, 0)
    values = {int(agent["agent_id"]): int(agent["input_shard"]) for agent in task["agent_configs"]}
    messages = []
    for path in sorted((ROOT / row["case_dir"]).glob("rounds/*/env/messages/*.json")):
        messages.append(json.loads(path.read_text(encoding="utf-8")))
    syntax_valid = 0
    faithful = 0
    for message in messages:
        value = values[int(message["sender_id"])]
        content = str(message["content"]).strip()
        decoded = decode_message(content, condition)
        syntax_valid += int(decoded is not None)
        faithful += int(decoded == value)
    return syntax_valid, faithful, len(messages)


def load_rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(RAW.open(newline="", encoding="utf-8")))
    for row in rows:
        task = json.loads((TASKS / row["task"]).read_text(encoding="utf-8"))
        gold = task["expected_output"]["per_agent_values"][0]
        submissions = json.loads(row["submissions"])
        row["input_bits"] = int(row["input_bits"])
        row["strict_correct"] = sum(bool(item.get("correct")) for item in submissions)
        row["semantic_correct"] = sum(
            answer_is_correct(item.get("answer"), gold) for item in submissions
        )
        row["numeric_submissions"] = sum(
            isinstance(item.get("answer"), int)
            and not isinstance(item.get("answer"), bool)
            for item in submissions
        )
        syntax_messages, faithful_messages, total_messages = message_format_adherence(row, task)
        row["syntax_valid_messages"] = syntax_messages
        row["faithful_messages"] = faithful_messages
        row["format_total_messages"] = total_messages
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
        "autoform",
        "no_communication",
    ]
    lines = [
        "# DuoSum v0.5 held-out comparison",
        "",
        f"**Status:** {len({row['task'] for row in rows})} held-out episodes, one greedy run per condition and episode, one local model. This is exploratory; it is not a confirmatory superiority comparison.",
        "",
        "## Pooled outcomes",
        "",
        "Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.",
        "",
        "| Condition | Strict agent success | Semantic exactness | Syntax-valid messages | Sender-value fidelity | Integer-only submissions | Mean total model tokens | Mean payload bytes | Mean messages | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        group = by_condition[condition]
        agent_count = 2 * len(group)
        strict = sum(row["strict_correct"] for row in group) / agent_count
        semantic = sum(row["semantic_correct"] for row in group) / agent_count
        numeric = sum(row["numeric_submissions"] for row in group) / agent_count
        syntax_messages = sum(row["syntax_valid_messages"] for row in group)
        faithful_messages = sum(row["faithful_messages"] for row in group)
        total_format_messages = sum(row["format_total_messages"] for row in group)
        syntax_rate = f"{syntax_messages}/{total_format_messages}" if total_format_messages else "n/a"
        fidelity_rate = f"{faithful_messages}/{total_format_messages}" if total_format_messages else "n/a"
        lines.append(
            f"| `{condition}` | {strict:.3f} | {semantic:.3f} | {syntax_rate} | {fidelity_rate} | {numeric:.3f} | "
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
            "- Compare communicating conditions with the observed no-communication control; do not assume the intervention worked from its label.",
            "- Syntax validity and sender-value fidelity are separate: the first asks whether a fixed-format message parses; the second asks whether it decodes to that sender's private value. They are not applicable to the unformatted scaffold, adaptive AutoForm, and no-communication controls.",
            "- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.",
            "- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.",
            "- This single-model, one-run-per-cell study is insufficient to estimate cross-model transfer, a scaling law, or a robust condition effect.",
            "- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.",
            "",
            "## Next revision",
            "",
            "Interpret outcomes jointly with format adherence. If fixed formats are followed and communication improves semantic task success, expand to a second receiver model and richer task family before matched-budget sweeps. If adherence remains poor, treat instruction-following as a bottleneck and avoid attributing outcomes to the intended representation.",
            "",
            "Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_5/`; this public summary contains aggregate measurements and selected anonymized outcomes only.",
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
