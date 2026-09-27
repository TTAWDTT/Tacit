"""Sanitize and summarize the frozen CRAFT v0.23 baseline traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import urllib.request


def tokenize(text: str, base_url: str) -> int:
    body = json.dumps({"content": text}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return len(json.load(response)["tokens"])


def llama_calls(path: Path, exclude: int) -> list[dict]:
    records: dict[str, dict] = {}
    task_re = re.compile(r"task\s+(\d+)\s+\|")
    prompt_re = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
    generation_re = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
    total_re = re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        task = task_re.search(line)
        if not task:
            continue
        record = records.setdefault(task.group(1), {"task_id": int(task.group(1))})
        if match := prompt_re.search(line):
            record.update(prompt_ms=float(match[1]), prompt_tokens=int(match[2]), prompt_tps=float(match[3]))
        elif match := generation_re.search(line):
            record.update(generation_ms=float(match[1]), generated_tokens=int(match[2]), generation_tps=float(match[3]))
        elif match := total_re.search(line):
            record.update(total_ms=float(match[1]), context_tokens=int(match[2]))
    complete = [
        record for record in records.values()
        if all(key in record for key in ["prompt_ms", "prompt_tokens", "prompt_tps", "generation_ms",
                                         "generated_tokens", "generation_tps", "total_ms", "context_tokens"])
    ]
    complete.sort(key=lambda record: record["task_id"])
    return complete[exclude:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--exclude-server-calls", type=int, default=58,
                        help="Completed calls from v0.21/v0.22 in the shared server log")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--tokenizer-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    expected = [
        ("structure_001", "medium"),
        ("structure_008", "simple"),
        ("structure_003", "complex"),
    ]
    games = {}
    for structure_id, complexity in expected:
        candidates = list(args.runs_dir.rglob(f"craft_{structure_id}_323.json"))
        if len(candidates) != 1:
            raise SystemExit(f"Expected one complete JSON result for {structure_id}; found {len(candidates)}")
        experiment = json.loads(candidates[0].read_text(encoding="utf-8"))
        game = experiment["games"][0]
        if game.get("complexity") != complexity:
            raise SystemExit(f"Unexpected structure level for {structure_id}: {game.get('complexity')}")
        games[structure_id] = game

    structures = []
    for structure_id, complexity in expected:
        game = games[structure_id]
        turns = []
        for turn in game.get("turns", []):
            responses = turn.get("director_responses", {})
            transcript = turn.get("director_discussion_full", "")
            delivered_roles = re.findall(r"(?m)^(D[123]): ", transcript)
            errors = {item.get("director") for item in turn.get("director_backend_errors", [])}
            messages = []
            for role in ["D1", "D2", "D3"]:
                response = responses.get(role)
                if not response:
                    continue
                public = response.get("public_message", "")
                successful = bool(response.get("raw_response")) and public not in ("", "No message provided")
                messages.append({
                    "role": role,
                    "backend_error": role in errors,
                    "successful_public_message": successful,
                    "delivered": role in delivered_roles,
                    "message": public if successful else "",
                    "message_tokens_with_role": tokenize(f"{role}: {public}", args.tokenizer_url)
                    if successful and role in delivered_roles else 0,
                })
            progress = turn.get("progress_data", {}).get("metrics", {}).get("overall_progress")
            move = turn.get("builder_move_raw", turn.get("progress_data", {}).get("move", {}))
            turns.append({
                "turn": turn.get("turn_number"),
                "director_slots_requested": 3,
                "successful_model_messages": sum(m["successful_public_message"] for m in messages),
                "backend_errors": len(errors),
                "builder_delivered_messages": sum(m["delivered"] for m in messages),
                "wire_transcript_tokens": tokenize(transcript, args.tokenizer_url) if transcript else 0,
                "progress": progress,
                "builder_followed_oracle": turn.get("builder_followed_oracle"),
                "selected_move": {k: move.get(k) for k in ["action", "block", "position", "layer", "span_to"] if k in move},
                "messages": messages,
            })
        structures.append({
            "id": structure_id,
            "complexity": complexity,
            "completed": game.get("completed"),
            "turns_taken": game.get("turns_taken"),
            "final_progress": game.get("final_progress"),
            "stopping_reason": game.get("stopping_reason"),
            "target_blocks": game.get("target_blocks"),
            "turns": turns,
        })

    calls = llama_calls(args.server_log, args.exclude_server_calls)
    all_turns = [turn for structure in structures for turn in structure["turns"]]
    all_messages = [message for turn in all_turns for message in turn["messages"]]
    successful_count = sum(message["successful_public_message"] for message in all_messages)
    result = {
        "study": "CRAFT bounded-history natural-language baseline v0.23",
        "source_revision": "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430",
        "model": "Qwen3-8B-Q4_K_M",
        "model_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        "condition": {"protocol": "CRAFT natural language", "history_window_lines": 16,
                       "server_context": 4096, "parallel_slots": 1},
        "structures": structures,
        "descriptive_means": {
            "final_progress": round(statistics.mean(s["final_progress"] for s in structures), 6),
            "successful_public_messages_per_requested": f"{successful_count}/{len(all_messages)}",
            "delivered_messages_per_successful": f"{sum(m['delivered'] for m in all_messages)}/{successful_count}",
            "wire_tokens_per_turn": round(statistics.mean(t["wire_transcript_tokens"] for t in all_turns), 2),
        },
        "inference": {
            "completed_server_calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "generated_tokens": sum(c["generated_tokens"] for c in calls),
            "service_seconds": round(sum(c["total_ms"] for c in calls) / 1000, 3),
            "mean_prompt_tokens_per_second": round(statistics.mean(c["prompt_tps"] for c in calls), 2) if calls else None,
            "mean_generation_tokens_per_second": round(statistics.mean(c["generation_tps"] for c in calls), 2) if calls else None,
            "calls": calls,
        },
        "limits": [
            "One model family, one run per structure, three structures, eight turns.",
            "The 16-line rolling shared history is part of this baseline condition.",
            "Oracle candidates are visible to the Builder; completion is not a test of free-form move search.",
            "The descriptive mean is not a population estimate or language superiority result.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CRAFT v0.23 bounded-history natural-language baseline",
        "",
        "This reference uses three difficulty levels and an explicit latest-16-line shared-history window. It is a baseline, not a protocol comparison.",
        "",
        "| Structure | Level | Turns | Final progress | Complete | Successful messages / requested | Delivered / successful | Mean wire tokens/turn |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for structure in structures:
        turns = structure["turns"]
        requested = sum(t["director_slots_requested"] for t in turns)
        successes = sum(t["successful_model_messages"] for t in turns)
        delivered = sum(t["builder_delivered_messages"] for t in turns)
        wire = statistics.mean(t["wire_transcript_tokens"] for t in turns) if turns else 0
        lines.append(f"| {structure['id']} | {structure['complexity']} | {structure['turns_taken']} | {structure['final_progress']:.3f} | {structure['completed']} | {successes}/{requested} | {delivered}/{successes} | {wire:.1f} |")
    inf = result["inference"]
    lines.extend([
        "",
        f"Across the three structures, descriptive mean progress was {result['descriptive_means']['final_progress']:.3f}. "
        f"The server completed {inf['completed_server_calls']} calls: {inf['prompt_tokens']} prompt tokens, "
        f"{inf['generated_tokens']} generated tokens, and {inf['service_seconds']} seconds of summed service time.",
        f"Mean prompt/generation throughput was {inf['mean_prompt_tokens_per_second']}/{inf['mean_generation_tokens_per_second']} tokens/s.",
        "",
        "## Scope",
        "",
        "This is a local capability and traffic reference under CRAFT's natural-language prompts and bounded shared history. It does not establish superiority, generalization, or an efficiency frontier. Future protocol conditions must reuse these tasks, model/runtime, history cap, scheduler, parser, and Builder oracle.",
        "",
        "Per-turn public messages, delivery counts, transcript token counts, and inference timing records are in the companion JSON. Private reasoning and prompts are excluded.",
        "",
    ])
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
