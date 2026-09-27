"""Analyze PrefixSum task success, role execution, message fidelity, and costs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".cache" / "pilot_v0_8" / "pilot_summary.csv"
TASKS = ROOT / "benchmarks" / "prefixsum_v0_2" / "tasks"
POLICIES = json.loads(
    (ROOT / "experiments" / "pilot_v0_8" / "policies.json").read_text(encoding="utf-8")
)
REPORT = ROOT / "research" / "PREFIXSUM_PILOT_V0_8.md"
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
        events = [
            json.loads(line)
            for log_path in (ROOT / row["case_dir"] / "logs").glob("agent-*.jsonl")
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        tool_calls = [event for event in events if event.get("event") == "tool_call"]
        tool_results = [event for event in events if event.get("event") == "tool_result"]
        row["agent0_send_to_1_actions"] = sum(
            event.get("agent_id") == 0
            and event.get("tool") == "send_message"
            and event.get("parameters", {}).get("target_id") == 1
            for event in tool_calls
        )
        row["agent1_send_actions"] = sum(
            event.get("agent_id") == 1 and event.get("tool") == "send_message"
            for event in tool_calls
        )
        successful_sends = [
            event for event in tool_results
            if event.get("agent_id") == 0
            and event.get("tool") == "send_message"
            and event.get("result", {}).get("success") is True
        ]
        row["agent0_successful_sends"] = len(successful_sends)
        nonempty_receives = [
            event for event in tool_results
            if event.get("agent_id") == 1
            and event.get("tool") == "receive_messages"
            and bool(event.get("result", {}).get("messages"))
        ]
        a1_submits = [
            event for event in tool_calls
            if event.get("agent_id") == 1 and event.get("tool") == "submit_result"
        ]
        row["agent1_received_payload_before_submit"] = int(
            bool(nonempty_receives) and bool(a1_submits)
            and min(event["timestamp"] for event in nonempty_receives)
            < min(event["timestamp"] for event in a1_submits)
        )
        syntax, faithful, messages = message_adherence(row, task)
        row["syntax"] = syntax
        row["faithful"] = faithful
        row["format_messages"] = messages
        for field in (
            "total_tokens", "message_payload_bytes", "message_count",
            "agent0_send_to_1_actions", "agent0_successful_sends",
            "agent1_received_payload_before_submit",
        ):
            row[field] = int(row[field])
        for field in (
            "agent0_send_to_1_actions",
            "agent1_send_actions",
            "agent0_send_before_submit",
            "agent1_receive_before_submit",
        ):
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
        "# PrefixSum v0.8 role-explicit protocol comparison",
        "",
        f"**Status:** {len({row['task'] for row in rows})} fresh seeded episodes, one greedy run per condition and episode, Qwen3-4B Q4_K_M on llama.cpp with a lossless message-content adapter. This is exploratory, not a confirmatory ranking.",
        "",
        "Strict agent success is exact equality of the returned integer list to that agent's pinned prefix-sum segment. Episode success requires both agents' lists to be exact. Message syntax and sender-value fidelity are separately audited from raw traces.",
        "",
        "## Pooled results",
        "",
        "| Condition | Strict agent success | Fully correct episodes | A0→A1 send attempts | Delivered messages | A1 received payload before submit | A1 send attempts | Message syntax | Sender-value fidelity | Mean payload bytes | Mean backend tokens | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"| `{name}` | {strict}/{n_agents} | {episode_success}/{len(group)} | "
            f"{sum(row['agent0_send_to_1_actions'] for row in group)} | "
            f"{sum(row['agent0_successful_sends'] for row in group)} | "
            f"{sum(row['agent1_received_payload_before_submit'] for row in group)}/{len(group)} | "
            f"{sum(row['agent1_send_actions'] for row in group)} | "
            f"{syntax_text} | {fidelity_text} | "
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
            "- Send attempts, successful delivery, and an Agent 1 receive call that returned a non-empty payload are distinct. `no_communication` still has send attempts because the intervention rejects the channel; it delivered zero messages. Agent 1's received-payload measure requires a non-empty receive result before its submit call.",
            "- Agent 0 sent a message with the required syntax in most format arms, but subtotal value fidelity was 0/12 for concise English, compact-KV, JSON, and binary. In contrast, full-shard syntax and value fidelity were both 12/12. This localizes a major failure to subtotal calculation or value encoding, before attributing downstream failures to receiver decoding.",
            "- Despite successful full-shard transmission, no episode was fully correct in any condition. The task therefore exposed a second bottleneck: correct receipt did not reliably produce Agent 1's offset-adjusted prefix list. In the audited compact-KV example, Agent 1 received `s=161` (the true sender shard sum was 168) and submitted only its local cumulative sums; the raw context shows the message was present. This is a sender aggregation and receiver execution failure, not transport loss.",
            "- The local adapter preserves the exact raw `send_message.content` string before the pinned Silo tool layer stores it. This avoids the upstream generic XML parser's integer/JSON coercion; it does not rewrite or normalize any model message.",
            "- For each segment length L and values in {1,...,50}, the exact subtotal has 49L+1 possible values. Any zero-error fixed-length encoding therefore needs at least ceil(log2(49L+1)) bits: 9, 10, and 11 bits at lengths 6, 15, and 30. This bound concerns the subtotal wire code, not model tokens, prompt cost, or the computation of the prefix arrays.",
            "- Compact-KV payloads averaged 5 bytes versus 64.2 bytes for full-shard payloads, but compact totals were never faithful in this run and did not yield episode success. This is only a wire-size comparison, not evidence of lower total cost or an efficiency-frontier gain.",
            "- One episode-condition run and twelve seeds provide exploratory evidence only. The two agent outputs in an episode are paired observations, not independent trials. No total-token budget is enforced and no efficiency frontier or scaling law is claimed.",
            "- Token counts are backend-reported. They are descriptive within this fixed setup and should not be compared directly across different model tokenizers or runtimes.",
            "",
            "## Next revision",
            "",
            "Before attempting another language comparison, repair the task execution bottleneck: provide a symbolic correctness check for the sender subtotal and an explicit receiver offset operation, then validate those mechanics with a non-LLM oracle and a small role-following control. Any follow-up should separately test raw subtotal calculation, wire decoding, and offset application; do not promote these results as a protocol ranking.",
            "",
            "Raw traces remain in ignored `.cache/pilot_v0_8/`; task files and checksums are public in `benchmarks/prefixsum_v0_2/tasks/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    print(build_report(rows), end="")
    REPORT.write_text(build_report(rows), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
