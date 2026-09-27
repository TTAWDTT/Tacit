"""Analyze v0.16 direct and simulator receiver traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
REPORT_JSON = ROOT / "research" / "PREFIXSUM_LOCAL_14B_RECEIVER_V0_16.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_LOCAL_14B_RECEIVER_V0_16.md"
PUBLIC = ROOT / "research" / "data" / "PREFIXSUM_LOCAL_14B_RECEIVER_V0_16_RUNS.jsonl"


def analyze(direct_path: Path, hybrid_path: Path) -> dict[str, Any]:
    direct = [json.loads(line) for line in direct_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hybrid = [json.loads(line) for line in hybrid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(direct) != 72 or len(hybrid) != 24:
        raise SystemExit(f"Expected 72 direct and 24 simulator episodes; found {len(direct)} and {len(hybrid)}")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in direct:
        groups[(row["condition"], int(row["segment_length"]))].append(row)
    direct_summary = []
    for (condition, length), rows in sorted(groups.items()):
        direct_summary.append({
            "condition": condition, "segment_length": length, "episodes": len(rows),
            "parse_success": sum(bool(row["parse_success"]) for row in rows),
            "exact": sum(bool(row["exact"]) for row in rows),
            "mean_input_tokens": round(sum(int(row["input_tokens"]) for row in rows) / len(rows), 1),
            "mean_output_tokens": round(sum(int(row["output_tokens"]) for row in rows) / len(rows), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in rows) / len(rows), 2),
        })
    hybrid_summary = []
    for length in (2, 3, 4):
        rows = [row for row in hybrid if int(row["segment_length"]) == length]
        received = [row for row in rows if row["agent1_received_payload_before_submit"]]
        hybrid_summary.append({
            "segment_length": length, "episodes": len(rows),
            "received_before_submit": len(received),
            "receiver_exact_given_receive": sum(float(row["success_rate"]) == 1.0 for row in received),
            "joint_exact": sum(float(row["success_rate"]) == 1.0 for row in rows),
            "message_syntax": sum(row["message_count"] > 0 for row in rows),
            "mean_backend_tokens": round(sum(int(row["total_tokens"]) for row in rows) / len(rows), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row in rows) / len(rows), 2),
        })
    baseline_v13 = json.loads((ROOT / "research" / "PREFIXSUM_ARITHMETIC_LADDER_V0_13.json").read_text(encoding="utf-8"))
    baseline_v11 = [json.loads(line) for line in (ROOT / "research" / "data" / "PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    base_hybrid = []
    for length in (2, 3, 4):
        rows = [row for row in baseline_v11 if row["model"] == "Qwen3-8B Q4_K_M GGUF" and row["variant"] == "oracle_sender_qwen_receiver" and int(row["segment_length"]) == length]
        received = [row for row in rows if row["agent1_received_payload_before_submit"]]
        base_hybrid.append({
            "segment_length": length, "episodes": len(rows),
            "received_before_submit": len(received),
            "receiver_exact_given_receive": sum(float(row["success_rate"]) == 1.0 for row in received),
            "joint_exact": sum(float(row["success_rate"]) == 1.0 for row in rows),
        })
    direct_base = [
        {"condition": row["condition"], "segment_length": row["segment_length"], "exact": row["exact"], "episodes": row["episodes"]}
        for row in baseline_v13["summaries_by_model_condition_length"] if row["model"] == "Qwen3-8B Q4_K_M GGUF"
    ]
    public_rows = []
    public_rows.extend(direct)
    for row in hybrid:
        run_dir = ROOT / row["case_dir"]
        messages = []
        for path in sorted(run_dir.glob("rounds/round-*/env/messages/*.json")):
            match = re.search(r"round-(\d+)", path.as_posix())
            message = json.loads(path.read_text(encoding="utf-8"))
            messages.append({"round": int(match.group(1)) if match else None, "sender_id": message.get("sender_id"), "recipient_id": message.get("recipient_id"), "content": message.get("content")})
        public_rows.append({
            "component": "simulator_receiver", "model": PREREG["model"], "condition": row["variant"],
            "task": row["task"], "segment_length": row["segment_length"],
            "success_rate": row["success_rate"], "submissions": json.loads(row["submissions"]),
            "messages": messages, "received_before_submit": row["agent1_received_payload_before_submit"],
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "elapsed_seconds": row["elapsed_seconds"], "rounds": row["rounds"],
        })
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_rows), encoding="utf-8", newline="\n")
    return {
        "study": PREREG["study"], "task_manifest_sha256": PREREG["task_manifest_sha256"],
        "model": PREREG["model"], "model_revision": PREREG["model_repository_revision"],
        "model_file_sha256": PREREG["model_file_sha256"], "runtime": PREREG["runtime"],
        "direct_summaries": direct_summary, "simulator_receiver_summaries": hybrid_summary,
        "qwen3_8b_v013_direct_baseline": direct_base,
        "qwen3_8b_v011_simulator_baseline": base_hybrid,
        "raw_trace_files": [str(direct_path.resolve().relative_to(ROOT)).replace("\\", "/"), str(hybrid_path.resolve().relative_to(ROOT)).replace("\\", "/")],
        "public_run_rows": "research/data/PREFIXSUM_LOCAL_14B_RECEIVER_V0_16_RUNS.jsonl",
        "limits": PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    receiver_exact = sum(row["joint_exact"] for row in result["simulator_receiver_summaries"])
    eligibility = (
        f"The preregistered eligibility gate was not met: {receiver_exact}/24 exact simulator receiver outputs, below the 20/24 threshold. "
        "Do not start the narrow message-format comparison under this task, receiver, runtime, and interface."
        if receiver_exact < 20 else
        f"The preregistered eligibility gate was met ({receiver_exact}/24 exact simulator receiver outputs). This only permits a separate, preregistered narrow message-format study; it is not evidence of protocol superiority."
    )
    lines = [
        "# Qwen3-14B local receiver capability v0.16", "",
        "A larger local checkpoint was evaluated on direct arithmetic controls and a true oracle-sender/model-receiver episode. Direct calls and simulator outcomes are reported separately.", "",
        f"Model revision `{result['model_revision']}`, GGUF SHA-256 `{result['model_file_sha256']}`. Task manifest SHA-256 `{result['task_manifest_sha256']}`. Runtime: {result['runtime']}.", "",
        "## Direct arithmetic ladder", "",
        "| Operation | L | Exact | Parseable | Mean input tokens | Mean output tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["direct_summaries"]:
        lines.append(f"| {row['condition']} | {row['segment_length']} | {row['exact']}/{row['episodes']} | {row['parse_success']}/{row['episodes']} | {row['mean_input_tokens']} | {row['mean_output_tokens']} |")
    lines.extend([
        "", "## Real simulator receiver condition", "",
        "| L | Correct message received before submit | Exact given receive | Joint exact |",
        "|---|---:|---:|---:|",
    ])
    for row in result["simulator_receiver_summaries"]:
        lines.append(f"| {row['segment_length']} | {row['received_before_submit']}/{row['episodes']} | {row['receiver_exact_given_receive']}/{row['received_before_submit']} | {row['joint_exact']}/{row['episodes']} |")
    lines.extend([
        "", "## Paired descriptive 8B references", "",
        "| L | 8B direct combined exact | 8B simulator receive | 8B exact given receive |",
        "|---|---:|---:|---:|",
    ])
    for length in (2, 3, 4):
        direct = next(row for row in result["qwen3_8b_v013_direct_baseline"] if row["condition"] == "combined_receiver" and row["segment_length"] == length)
        simulator = next(row for row in result["qwen3_8b_v011_simulator_baseline"] if row["segment_length"] == length)
        lines.append(f"| {length} | {direct['exact']}/{direct['episodes']} | {simulator['received_before_submit']}/{simulator['episodes']} | {simulator['receiver_exact_given_receive']}/{simulator['received_before_submit']} |")
    lines.extend([
        "", "## Interpretation", "",
        "Interpret only the direct operation ladder and real simulator receiver path separately. " + eligibility + " Any model-to-model differences remain confounded by checkpoint, scale, quantization, and CPU/GPU offload.",
        "",
        "This is one held-out case family reused from earlier pilots, one greedy sample per case, and a local mixed-offload runtime. No communication-efficiency or general-language claim follows.",
        "",
        f"Per-episode prompts, exact arrays, messages, answers, and costs: [JSONL]({Path('data/PREFIXSUM_LOCAL_14B_RECEIVER_V0_16_RUNS.jsonl').as_posix()}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-runs", type=Path, required=True)
    parser.add_argument("--hybrid-runs", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.direct_runs, args.hybrid_runs)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(REPORT_MD)


if __name__ == "__main__":
    main()
