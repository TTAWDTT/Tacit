"""Analyze PrefixSum task success, message fidelity, and costs for v0.7."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".cache" / "pilot_v0_7" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "prefixsum_v0_1" / "tasks"
POLICIES = json.loads(
    (ROOT / "experiments" / "pilot_v0_7" / "policies.json").read_text(encoding="utf-8")
)
REPORT = ROOT / "research" / "PREFIXSUM_PILOT_V0_7.md"
CONDITIONS = [
    "scaffold_only",
    "concise_nl",
    "compact_kv",
    "json_schema",
    "binary",
    "full_shard",
    "no_communication",
]


def message_adherence(row: dict[str, Any], task: dict[str, Any]) -> tuple[int, int, int]:
    condition = row["condition"]
    if condition in {"scaffold_only", "no_communication"}:
        return 0, 0, 0
    shards = {int(agent["agent_id"]): agent["input_shard"] for agent in task["agent_configs"]}
    syntax = faithful = total = 0
    case_dir = ROOT / row["case_dir"]
    for path in sorted(case_dir.glob("rounds/*/env/messages/*.json")):
        message = json.loads(path.read_text(encoding="utf-8"))
        content = str(message["content"]).strip()
        sender = int(message["sender_id"])
        expected = shards[sender]
        decoded: Any = None
        if condition == "concise_nl":
            match = re.fullmatch(r"The subtotal is (\d+)\.", content)
            decoded = int(match.group(1)) if match else None
        elif condition == "compact_kv":
            match = re.fullmatch(r"s=(\d+)", content)
            decoded = int(match.group(1)) if match else None
        elif condition == "json_schema":
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and set(obj) == {"s"} and type(obj["s"]) is int:
                decoded = obj["s"]
        elif condition == "binary":
            if re.fullmatch(r"[01]+", content):
                value = int(content, 2)
                if content == format(value, "b"):
                    decoded = value
        elif condition == "full_shard":
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                decoded = None
        if condition == "full_shard":
            expected_value: Any = expected
            expected_syntax = isinstance(decoded, list) and all(type(v) is int for v in decoded)
        else:
            expected_value = sum(expected)
            expected_syntax = decoded is not None
        syntax += int(expected_syntax)
        faithful += int(decoded == expected_value)
        total += 1
    return syntax, faithful, total


def load_rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(RAW.open(newline="", encoding="utf-8")))
    for row in rows:
        task = json.loads((TASKS / row["task"]).read_text(encoding="utf-8"))
        results_path = ROOT / row["case_dir"] / "results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        submissions = json.loads(row["submissions"])
        row["strict_correct"] = sum(bool(item.get("correct")) for item in submissions)
        row["episode_success"] = int(results["metrics"]["S_success_rate"] == 1.0)
        syntax, faithful, messages = message_adherence(row, task)
        row["syntax"] = syntax
        row["faithful"] = faithful
        row["format_messages"] = messages
        for field in ("total_tokens", "message_payload_bytes", "message_count"):
            row[field] = int(row[field])
        row["segment_length"] = int(row["segment_length"])
        row["elapsed_seconds"] = float(row["elapsed_seconds"])
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def build_report(rows: list[dict[str, Any]]) -> str:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_length: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_length[(row["segment_length"], row["condition"])].append(row)

    lines = [
        "# PrefixSum v0.7 held-out protocol comparison",
        "",
        f"**Status:** {len({row['task'] for row in rows})} deterministic held-out episodes, one greedy run per condition and episode, Qwen3-4B Q4_K_M on llama.cpp. This is exploratory, not a confirmatory ranking.",
        "",
        "Strict agent success is exact equality of the returned integer list to that agent's pinned prefix-sum segment. Episode success requires both agents' lists to be exact. Message syntax and sender-value fidelity are separately audited from raw traces.",
        "",
        "## Pooled results",
        "",
        "| Condition | Strict agent success | Fully correct episodes | Message syntax | Sender-value fidelity | Mean payload bytes | Mean backend tokens | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONDITIONS:
        group = by_condition[name]
        n_agents = 2 * len(group)
        strict = sum(row["strict_correct"] for row in group)
        episode_success = sum(row["episode_success"] for row in group)
        n_messages = sum(row["format_messages"] for row in group)
        syntax = sum(row["syntax"] for row in group)
        fidelity = sum(row["faithful"] for row in group)
        syntax_text = f"{syntax}/{n_messages}" if n_messages else "n/a"
        fidelity_text = f"{fidelity}/{n_messages}" if n_messages else "n/a"
        lines.append(
            f"| `{name}` | {strict}/{n_agents} | {episode_success}/{len(group)} | {syntax_text} | {fidelity_text} | "
            f"{mean(group, 'message_payload_bytes'):.1f} | {mean(group, 'total_tokens'):.0f} | {mean(group, 'elapsed_seconds'):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Results by segment length",
            "",
            "| Segment length | Condition | Strict agent success | Fully correct episodes | Mean payload bytes | Mean backend tokens |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for length in sorted({row["segment_length"] for row in rows}):
        for name in CONDITIONS:
            group = by_length[(length, name)]
            n_agents = 2 * len(group)
            lines.append(
                f"| {length} | `{name}` | {sum(row['strict_correct'] for row in group)}/{n_agents} | "
                f"{sum(row['episode_success'] for row in group)}/{len(group)} | "
                f"{mean(group, 'message_payload_bytes'):.1f} | {mean(group, 'total_tokens'):.0f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare each communicating arm with the no-communication control. Agent 0 can compute its local prefix segment without receiving Agent 1's values, but Agent 1 cannot recover its global offset from its own segment alone; therefore episode-level unanimity is the task-necessity outcome.",
            "- For each segment length L and values in {1,...,50}, the exact subtotal has 49L+1 possible values. Any zero-error fixed-length encoding therefore needs at least ceil(log2(49L+1)) bits: 9, 10, and 11 bits at lengths 6, 15, and 30. This bound concerns the subtotal wire code, not model tokens, prompt cost, or the computation of the prefix arrays.",
            "- Compare compact subtotal forms with `full_shard` to test whether transmitting a sufficient statistic preserves success while reducing payload bytes. Failures can arise from subtotal calculation, encoding/decoding, or prefix-list computation; the aggregate score alone cannot separate these stages.",
            "- One episode-condition run and twelve seeds provide exploratory evidence only. The two agent outputs in an episode are paired observations, not independent trials. No total-token budget is enforced and no efficiency frontier or scaling law is claimed.",
            "- Token counts are backend-reported. They are descriptive within this fixed setup and should not be compared directly across different model tokenizers or runtimes.",
            "",
            "## Next revision",
            "",
            "If compact subtotal messages remain accurate and smaller than full-shard transfer, increase segment length and agent count while varying total communication budget. If errors occur, classify arithmetic aggregation, representation syntax, receiver decoding, and array construction separately before changing the protocol.",
            "",
            "Raw traces remain in ignored `.cache/pilot_v0_7/`; task files and checksums are public in `benchmarks/prefixsum_v0_1/tasks/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    print(build_report(rows), end="")
    REPORT.write_text(build_report(rows), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
