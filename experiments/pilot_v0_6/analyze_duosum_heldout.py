"""Analyze v0.6 DuoSum outcomes, serialization, and fixed-format fidelity."""

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


RAW = ROOT / ".cache" / "pilot_v0_6" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "duosum_v0_1" / "tasks"
MANIFEST = json.loads((TASKS / "manifest.json").read_text(encoding="utf-8"))
MANIFEST_BY_FILE = {entry["file"]: entry for entry in MANIFEST["files"]}
POLICIES = json.loads((ROOT / "experiments" / "pilot_v0_6" / "policies.json").read_text(encoding="utf-8"))
REPORT = ROOT / "research" / "DUOSUM_PILOT_V0_6.md"


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
        sender_values = {int(agent["agent_id"]): int(agent["input_shard"]) for agent in task["agent_configs"]}
        bare_decimal_matches = 0
        bare_decimal_total = 0
        for path in sorted((ROOT / row["case_dir"]).glob("rounds/*/env/messages/*.json")):
            message = json.loads(path.read_text(encoding="utf-8"))
            content = str(message["content"]).strip()
            if content.isdecimal():
                bare_decimal_total += 1
                bare_decimal_matches += int(
                    int(content) == sender_values[int(message["sender_id"])]
                )
        row["bare_decimal_matches"] = bare_decimal_matches
        row["bare_decimal_total"] = bare_decimal_total
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
        "# DuoSum v0.6 cross-model held-out replication",
        "",
        f"**Status:** {len({row['task'] for row in rows})} held-out episodes, one greedy run per condition and episode, Qwen3-4B Q4_K_M on llama.cpp. This is exploratory; it is not a confirmatory superiority comparison.",
        "",
        "## Pooled outcomes",
        "",
        "Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.",
        "",
        "| Condition | Strict agent success | Semantic exactness | Assigned-grammar syntax | Grammar-decoded value fidelity | Bare-decimal value matches | Integer-only submissions | Mean total model tokens | Mean payload bytes | Mean messages | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        bare_total = sum(row["bare_decimal_total"] for row in group)
        bare_match = sum(row["bare_decimal_matches"] for row in group)
        bare_rate = f"{bare_match}/{bare_total}" if bare_total else "n/a"
        lines.append(
            f"| `{condition}` | {strict:.3f} | {semantic:.3f} | {syntax_rate} | {fidelity_rate} | {bare_rate} | {numeric:.3f} | "
            f"{mean(group, 'total_tokens'):.1f} | {mean(group, 'message_payload_bytes'):.1f} | "
            f"{mean(group, 'message_count'):.2f} | {mean(group, 'elapsed_seconds'):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-width results",
            "",
            "Rates are over four agent submissions from two episodes at each width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.",
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

    v05_strict = {
        "scaffold_only": 4,
        "concise_nl": 3,
        "compact_kv": 0,
        "json_schema": 3,
        "binary": 5,
        "autoform": 4,
        "no_communication": 1,
    }
    v05_semantic = {
        "scaffold_only": 15,
        "concise_nl": 11,
        "compact_kv": 10,
        "json_schema": 4,
        "binary": 13,
        "autoform": 12,
        "no_communication": 1,
    }
    lines.extend(
        [
            "",
            "## Paired comparison with v0.5",
            "",
            "The eight held-out task files and seven conditions are the same as v0.5. Each column counts correct agent outputs out of 16; the two agents within one episode are not independent samples. This is a descriptive paired replication across two model setups, not an isolated model-scale effect.",
            "",
            "| Condition | v0.5 strict | v0.6 strict | Difference | v0.5 semantic | v0.6 semantic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        group = by_condition[condition]
        current_strict = sum(row["strict_correct"] for row in group)
        current_semantic = sum(row["semantic_correct"] for row in group)
        lines.append(
            f"| `{condition}` | {v05_strict[condition]}/16 | {current_strict}/16 | "
            f"{current_strict - v05_strict[condition]:+d} | {v05_semantic[condition]}/16 | {current_semantic}/16 |"
        )

    total_agents = 2 * len(by_condition["scaffold_only"])
    semantic_rates = {
        name: sum(row["semantic_correct"] for row in by_condition[name])
        for name in conditions
    }
    no_comm_semantic = semantic_rates["no_communication"]
    communicating_semantic = [semantic_rates[name] for name in conditions if name != "no_communication"]
    binary_messages = sum(row["format_total_messages"] for row in by_condition["binary"])
    binary_faithful = sum(row["faithful_messages"] for row in by_condition["binary"])
    json_messages = sum(row["format_total_messages"] for row in by_condition["json_schema"])
    json_syntax = sum(row["syntax_valid_messages"] for row in by_condition["json_schema"])
    kv_messages = sum(row["format_total_messages"] for row in by_condition["compact_kv"])
    kv_faithful = sum(row["faithful_messages"] for row in by_condition["compact_kv"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare communicating conditions with the observed no-communication control; do not assume the intervention worked from its label.",
            "- Assigned-grammar syntax and grammar-decoded value fidelity apply to fixed-format arms. The separate bare-decimal column records whether a decimal-only message equals its sender's private value; it does not count as compliance with a sentence or schema instruction.",
            "- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.",
            f"- No-communication semantic success was {no_comm_semantic}/{total_agents} agent outputs. Communicating arms ranged from {min(communicating_semantic)}/{total_agents} to {max(communicating_semantic)}/{total_agents}. The zero control score is consistent with communication being necessary on these cases, but eight episodes are not a proof for the full task family.",
            f"- The concise-NL instruction specified a plain-English sentence, but the raw audit shows decimal-only payloads in {sum(row['bare_decimal_total'] for row in by_condition['concise_nl'])}/{sum(row['format_total_messages'] for row in by_condition['concise_nl'])} messages. Those numerals matched their senders' private values in {sum(row['bare_decimal_matches'] for row in by_condition['concise_nl'])}/{sum(row['bare_decimal_total'] for row in by_condition['concise_nl'])}; this is effective decimal shorthand, not adherence to the assigned NL form.",
            f"- Compact-KV messages decoded to the sender's value in {kv_faithful}/{kv_messages} messages, yet semantic task success was {semantic_rates['compact_kv']}/{total_agents}. This separates reliable serialization from downstream reasoning/submission failures.",
            f"- The example-assisted binary arm encoded the sender's value correctly in only {binary_faithful}/{binary_messages} messages; JSON syntax was valid in {json_syntax}/{json_messages} messages. Raw audit found all 16 JSON messages were single-quoted Python-style mappings, so the 16/16 strict task success reflects receiver tolerance of that payload rather than valid-JSON adherence.",
            "- Within this run, concise-NL's actual decimal-only payload averaged 6 bytes and reached 14/16 strict successes; compact-KV averaged 10 bytes and reached 15/16; the invalid-JSON mapping averaged 20 bytes and reached 16/16. This is a small descriptive trade-off, not a matched-budget frontier, and the first and third arms did not follow their assigned grammars.",
            f"- AutoForm semantic success was {semantic_rates['autoform']}/{total_agents}, compared with {semantic_rates['scaffold_only']}/{total_agents} for the unformatted scaffold. This small run shows no evidence that format self-selection improves task success for this model/task.",
            "- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.",
            "- Relative to v0.5, strict success increased for scaffold (+3 agents), concise NL (+11), compact KV (+15), JSON (+13), and AutoForm (+7); binary fell by 2 and no-communication fell by 1. This paired pattern is exploratory: model size, quantization, backend, and chat template all change together, so it cannot isolate scale or establish a general condition effect.",
            "- Token totals are reported as returned by each backend. Do not compare v0.5 and v0.6 model-token totals as a cross-model efficiency result because the model tokenizer and inference backend changed.",
            "- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.",
            "",
            "## Next revision",
            "",
            "Next, evaluate a typed parser/decoder as an explicit system condition against prompt-only formatting, then replicate across a richer task family and additional sender/receiver pairs. Account for parser cost, malformed-message recovery, and wire size. Do not call the invalid JSON arm JSON adherence or compare backend token totals directly across v0.5/v0.6.",
            "",
            "Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_6/`; this public summary contains aggregate measurements and selected anonymized outcomes only.",
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
