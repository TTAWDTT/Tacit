"""Rebuild the v0.10 report from raw traces; makes no model calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import run_scale_hybrid as study
import run_prefixsum as runner
from analyze_prefixsum import message_adherence

ROOT = study.ROOT
REPORT_JSON = ROOT / "research" / "PREFIXSUM_MODEL_SCALE_V0_10.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_MODEL_SCALE_V0_10.md"


def local_prefix(values: list[int]) -> list[int]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def episode(row: dict[str, Any]) -> dict[str, Any]:
    task = runner.read_json(runner.TASKS / row["task"])
    shards = [item["input_shard"] for item in task["agent_configs"]]
    expected = task["expected_output"]["per_agent_values"]
    submissions = {int(item["agent_id"]): item for item in json.loads(row["submissions"])}
    messages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / row["case_dir"]).glob("rounds/round-*/env/messages/*.json"))
    ]
    message = messages[0] if messages else None
    match = re.fullmatch(r"s=(\d+)", str(message["content"])) if message else None
    wire = int(match.group(1)) if match else None
    a1 = submissions.get(1, {}).get("answer")
    lp = local_prefix(shards[1])
    wire_followed = wire is not None and a1 == [wire + item for item in lp]
    return {
        "agent0_exact": bool(submissions.get(0, {}).get("correct")),
        "agent1_exact": bool(submissions.get(1, {}).get("correct")),
        "joint_exact": float(row["success_rate"]) == 1.0,
        "agent1_received_before_submit": bool(row["agent1_received_payload_before_submit"]),
        "agent1_local_prefix_only": a1 == lp,
        "agent1_followed_wire": wire_followed,
        "message_syntax": match is not None,
        "message_faithful": wire == sum(shards[0]),
        "segment_length": int(task["metadata"]["segment_length"]),
    }


def analyze(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = len(runner.TASK_FILES) * len(study.VARIANTS)
    if len(rows) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} model-scale episodes, found {len(rows)}")
    by_variant = []
    for name in study.VARIANTS:
        group = [row for row in rows if row["variant"] == name]
        records = [episode(row) for row in group]
        received = [record for record in records if record["agent1_received_before_submit"]]
        syntax, fidelity = 0, 0
        for row in group:
            task = runner.read_json(runner.TASKS / row["task"])
            s, f, _ = message_adherence(row, task)
            syntax += s
            fidelity += f
        by_variant.append(
            {
                "variant": name,
                "episodes": len(group),
                "agent0_exact": sum(item["agent0_exact"] for item in records),
                "agent1_exact": sum(item["agent1_exact"] for item in records),
                "fully_correct_episodes": sum(item["joint_exact"] for item in records),
                "agent1_received_before_submit": len(received),
                "agent1_exact_given_received": sum(item["agent1_exact"] for item in received),
                "agent1_local_prefix_only": sum(item["agent1_local_prefix_only"] for item in records),
                "agent1_followed_wire_value": sum(item["agent1_followed_wire"] for item in records),
                "message_syntax": syntax,
                "subtotal_value_fidelity": fidelity,
                "delivered_messages": sum(int(row["message_count"]) for row in group),
                "mean_backend_tokens": round(sum(int(row["total_tokens"]) for row in group) / len(group), 1),
                "mean_payload_bytes": round(sum(int(row["message_payload_bytes"]) for row in group) / len(group), 1),
                "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in group) / len(group), 2),
            }
        )
    baseline = json.loads((ROOT / "research" / "PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.json").read_text(encoding="utf-8"))
    return {
        "study": "local_model_scale_hybrid_diagnostic_v0_10",
        "comparison_baseline": "paired Qwen3-4B v0.9 on the same 12 tasks",
        "model": study.PREREG["model"],
        "model_repository_revision": study.PREREG["model_repository_revision"],
        "model_file_sha256": study.PREREG["model_file_sha256"],
        "task_manifest_sha256": study.PREREG["task_manifest_sha256"],
        "runtime": study.PREREG["runtime"],
        "variants": by_variant,
        "qwen3_4b_baseline": baseline["variants"],
        "limits": study.PREREG["limits"],
        "raw_trace_file": f".cache/pilot_v0_10/{path.name}",
    }


def markdown(result: dict[str, Any]) -> str:
    v8 = {row["variant"]: row for row in result["variants"]}
    v4 = {row["variant"]: row for row in result["qwen3_4b_baseline"]}
    fields = [
        ("Agent 0 exact", "agent0_exact"),
        ("Agent 1 exact given receive", "agent1_exact_given_received"),
        ("Fully correct episodes", "fully_correct_episodes"),
        ("Correct subtotal messages", "subtotal_value_fidelity"),
    ]
    lines = [
        "# PrefixSum v0.10 local model-scale diagnostic",
        "",
        "**Design:** Qwen3-8B Q4_K_M in the two preregistered oracle/model hybrid roles, compared with Qwen3-4B v0.9 on the same 12 tasks and compact-KV wire condition. The task set is reused, and the 4B values are the paired v0.9 results; this is not new held-out evidence or a language ranking.",
        "",
        f"Model revision `{result['model_repository_revision']}`, file SHA-256 `{result['model_file_sha256']}`. Runtime: {result['runtime']}.",
        "",
    ]
    for role, variant in (
        ("Oracle sender + model receiver", "oracle_sender_qwen_receiver"),
        ("Model sender + oracle receiver", "qwen_sender_oracle_receiver"),
    ):
        lines.extend([f"## {role}", "", "| Metric | Qwen3-4B | Qwen3-8B |", "|---|---:|---:|"])
        for label, key in fields:
            if key == "agent1_exact_given_received":
                four = f"{v4[variant]['agent1_exact_given_received']}/{v4[variant]['agent1_received_before_submit']}"
                eight = f"{v8[variant]['agent1_exact_given_received']}/{v8[variant]['agent1_received_before_submit']}"
            else:
                four = f"{v4[variant][key]}/12"
                eight = f"{v8[variant][key]}/12"
            lines.append(f"| {label} | {four} | {eight} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Compare role-specific accuracy and message fidelity with the preregistered prediction; do not interpret token or latency differences as protocol efficiency. A model-size comparison also changes the checkpoint and its training, so it cannot isolate parameter count alone. One greedy draw on 12 reused cases supports only a local diagnostic.",
            "",
            f"Raw local traces: `{result['raw_trace_file']}`. Aggregate data: [JSON](PREFIXSUM_MODEL_SCALE_V0_10.json).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    path = args.runs or max((ROOT / ".cache" / "pilot_v0_10").glob("scale_hybrid_runs_*.jsonl"), key=lambda item: item.stat().st_mtime)
    result = analyze(path)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
