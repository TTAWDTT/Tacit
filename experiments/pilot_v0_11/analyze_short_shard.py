"""Analyze v0.11 raw traces and publish per-episode rows; makes no model calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_short_shard as study
import run_prefixsum as runner

ROOT = study.ROOT
REPORT_JSON = ROOT / "research" / "PREFIXSUM_SHORT_SHARD_V0_11.json"
REPORT_MD = ROOT / "research" / "PREFIXSUM_SHORT_SHARD_V0_11.md"
RUN_ROWS = ROOT / "research" / "data" / "PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl"


def local_prefix(values: list[int]) -> list[int]:
    result, total = [], 0
    for value in values:
        total += value
        result.append(total)
    return result


def episode(row: dict[str, Any]) -> dict[str, Any]:
    task = runner.read_json(runner.TASKS / row["task"])
    shards = [item["input_shard"] for item in task["agent_configs"]]
    submissions = {int(item["agent_id"]): item for item in json.loads(row["submissions"])}
    messages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / row["case_dir"]).glob("rounds/round-*/env/messages/*.json"))
    ]
    message = messages[0] if messages else None
    match = re.fullmatch(r"s=(\d+)", str(message["content"])) if message else None
    wire = int(match.group(1)) if match else None
    a0 = submissions.get(0, {}).get("answer")
    a1 = submissions.get(1, {}).get("answer")
    expected = task["expected_output"]["per_agent_values"]
    return {
        "agent0_exact": a0 == expected[0],
        "agent1_exact": a1 == expected[1],
        "joint_exact": float(row["success_rate"]) == 1.0,
        "agent1_received_payload_before_submit": bool(row["agent1_received_payload_before_submit"]),
        "agent1_exact_given_received": a1 == expected[1] and bool(row["agent1_received_payload_before_submit"]),
        "agent1_local_prefix_only": a1 == local_prefix(shards[1]),
        "message_syntax": match is not None,
        "message_faithful": wire == sum(shards[0]),
        "segment_length": int(task["metadata"]["segment_length"]),
        "seed": int(task["metadata"]["seed"]),
        "wire_subtotal": wire,
        "expected_subtotal": sum(shards[0]),
        "submissions": {str(k): v.get("answer") for k, v in submissions.items()},
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = len(runner.TASK_FILES) * len(study.VARIANTS) * len(paths)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} episodes across traces, found {len(rows)}")
    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        task = runner.read_json(runner.TASKS / row["task"])
        record = episode(row)
        grouped[(row["model"], row["variant"], record["segment_length"])].append((row, record))
    summaries = []
    for (model, variant, length), group in sorted(grouped.items()):
        records = [record for _, record in group]
        receiver_arm = variant == "oracle_sender_qwen_receiver"
        received = [r for r in records if r["agent1_received_payload_before_submit"]] if receiver_arm else []
        summaries.append({
            "model": model,
            "variant": variant,
            "segment_length": length,
            "episodes": len(group),
            "agent0_exact": sum(r["agent0_exact"] for r in records),
            "agent1_exact": sum(r["agent1_exact"] for r in records),
            "joint_exact": sum(r["joint_exact"] for r in records),
            "message_syntax": sum(r["message_syntax"] for r in records),
            "subtotal_fidelity": sum(r["message_faithful"] for r in records),
            "receiver_received_before_submit": len(received),
            "receiver_exact_given_received": sum(r["agent1_exact"] for r in received),
            "receiver_local_prefix_only": sum(r["agent1_local_prefix_only"] for r in records),
            "mean_backend_tokens": round(sum(int(row["total_tokens"]) for row, _ in group) / len(group), 1),
            "mean_payload_bytes": round(sum(int(row["message_payload_bytes"]) for row, _ in group) / len(group), 1),
            "mean_wall_seconds": round(sum(float(row["elapsed_seconds"]) for row, _ in group) / len(group), 2),
        })
    public = []
    for row in rows:
        record = episode(row)
        run_dir = ROOT / row["case_dir"]
        messages = []
        for path in sorted(run_dir.glob("rounds/round-*/env/messages/*.json")):
            match = re.search(r"round-(\d+)", path.as_posix())
            data = json.loads(path.read_text(encoding="utf-8"))
            messages.append({"round": int(match.group(1)) if match else None, "sender_id": data.get("sender_id"), "recipient_id": data.get("recipient_id"), "content": data.get("content")})
        public.append({
            "model": row["model"], "variant": row["variant"], "task": row["task"],
            "seed": record["seed"], "segment_length": record["segment_length"],
            "success_rate": row["success_rate"], "submissions": record["submissions"],
            "messages": messages, "message_syntax": record["message_syntax"],
            "subtotal_fidelity": record["message_faithful"],
            "agent1_received_payload_before_submit": record["agent1_received_payload_before_submit"],
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "message_payload_bytes": row["message_payload_bytes"],
            "elapsed_seconds": row["elapsed_seconds"], "rounds": row["rounds"],
        })
    RUN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    RUN_ROWS.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in public), encoding="utf-8", newline="\n")
    return {
        "study": study.PREREG["study"], "task_manifest_sha256": study.PREREG["task_manifest_sha256"],
        "models": study.PREREG["conditions"], "runtime": study.PREREG["runtime"],
        "summaries_by_role_and_length": summaries,
        "raw_trace_files": [str(path.resolve().relative_to(ROOT)) for path in paths],
        "public_run_rows": "research/data/PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl",
        "limits": study.PREREG["limits"],
    }


def markdown(result: dict[str, Any]) -> str:
    summaries = result["summaries_by_role_and_length"]
    sender = [item for item in summaries if item["variant"] == "qwen_sender_oracle_receiver"]
    receiver = [item for item in summaries if item["variant"] == "oracle_sender_qwen_receiver"]
    lines = [
        "# PrefixSum v0.11 short-shard role-capability calibration", "",
        "This preregistered study measures sender/receiver capability at held-out segment lengths 2, 3, and 4 under one compact-KV message condition. It does not rank protocols.", "",
        f"Task manifest SHA-256: `{result['task_manifest_sha256']}`. Runtime: {result['runtime']}.", "",
        "Each cell has 8 episodes. Oracle outputs are excluded from model capability counts. In the receiver arm, the oracle sender emits the correct subtotal; a receive counts only if Agent 1 gets a nonempty payload before submitting.", "",
        "## Model sender + oracle receiver", "",
        "| Model | L | Qwen Agent 0 exact | Subtotal faithful | Joint exact | Message syntax |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in sender:
        lines.append(f"| {item['model']} | {item['segment_length']} | {item['agent0_exact']}/{item['episodes']} | {item['subtotal_fidelity']}/{item['episodes']} | {item['joint_exact']}/{item['episodes']} | {item['message_syntax']}/{item['episodes']} |")
    lines.extend([
        "", "## Model receiver + oracle sender", "",
        "| Model | L | Correct oracle messages received before submit | Qwen exact given receive | Qwen local-prefix-only output |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in receiver:
        lines.append(f"| {item['model']} | {item['segment_length']} | {item['receiver_received_before_submit']}/{item['episodes']} | {item['receiver_exact_given_received']}/{item['receiver_received_before_submit']} | {item['receiver_local_prefix_only']}/{item['episodes']} |")
    recv_totals = {}
    for model in {item["model"] for item in receiver}:
        rows = [item for item in receiver if item["model"] == model]
        recv_totals[model] = (sum(i["receiver_received_before_submit"] for i in rows), sum(i["receiver_exact_given_received"] for i in rows))
    four = next((v for k, v in recv_totals.items() if "4B" in k), (0, 0))
    eight = next((v for k, v in recv_totals.items() if "8B" in k), (0, 0))
    lines.extend([
        "", "## Interpretation", "",
        f"Short shards exposed a sender-side operating region, but not a two-model communication region. At L=2, the model sender plus oracle receiver completed {next(i['joint_exact'] for i in sender if '4B' in i['model'])}/8 Qwen3-4B episodes and {next(i['joint_exact'] for i in sender if '8B' in i['model'])}/8 Qwen3-8B episodes. Sender subtotal fidelity and Agent 0's own output are separate: a faithful message alone is not a joint success.",
        "",
        f"On the receiver side, Qwen3-4B received no payload before submission in {sum(i['episodes'] for i in receiver if '4B' in i['model'])} episodes. Qwen3-8B received the correct oracle message in {eight[0]}/24 episodes, but returned an exact global segment in {eight[1]}/{eight[0]} received-message cases. Thus shorter inputs did not resolve receiver-side tool use/integration in this setup. The receiver result combines model behavior with the pinned simulator and interaction loop; it does not isolate arithmetic alone.",
        "",
        "No tested length supports a protocol comparison yet because the model receiver never completed the task, despite correct oracle inputs in the cases where it received them. Next, isolate message acquisition from message application with a preregistered direct-context control, keeping the current task suite and exact-output scorer. Do not interpret these outcomes as evidence that compact-KV is more efficient or that a new language is needed.",
        "",
        "Eight cases per cell and one greedy run do not support significance claims or decoding-variance estimates. This is one task family, one local engine, and two checkpoint/quantization configurations.",
        "",
        f"Per-episode submissions and exact wire messages: [JSONL run rows]({Path('data/PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl').as_posix()}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    # Configure task paths without checking model files or making any inference calls.
    runner.TASKS = ROOT / "benchmarks" / "prefixsum_v0_3" / "tasks"
    manifest = json.loads((runner.TASKS / "manifest.json").read_text(encoding="utf-8"))
    runner.TASK_FILES = [entry["file"] for entry in manifest["files"]]
    result = analyze(args.runs)
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_MD.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(REPORT_MD)


if __name__ == "__main__":
    main()
