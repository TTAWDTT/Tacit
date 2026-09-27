"""Create a safe, reproducible CRAFT v0.21 aggregate without private prompts/CoT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import urllib.error
import urllib.request


def tokenize(text: str, base_url: str) -> int:
    body = json.dumps({"content": text}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            tokens = json.load(response)["tokens"]
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        raise SystemExit(f"Could not tokenize the preserved public transcript: {exc}") from exc
    return len(tokens)


def server_calls(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records: dict[str, dict] = {}
    task_re = re.compile(r"task\s+(\d+)\s+\|")
    prompt_re = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
    generation_re = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
    total_re = re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens")
    for line in text.splitlines():
        task = task_re.search(line)
        if not task:
            continue
        task_id = task.group(1)
        record = records.setdefault(task_id, {"task_id": int(task_id)})
        prompt = prompt_re.search(line)
        generation = generation_re.search(line)
        total = total_re.search(line)
        if prompt:
            record.update(
                prompt_ms=float(prompt.group(1)),
                prompt_tokens=int(prompt.group(2)),
                prompt_tokens_per_second=float(prompt.group(3)),
            )
        elif generation:
            record.update(
                generation_ms=float(generation.group(1)),
                generated_tokens=int(generation.group(2)),
                generation_tokens_per_second=float(generation.group(3)),
            )
        elif total:
            record.update(total_ms=float(total.group(1)), context_tokens=int(total.group(2)))
    return [
        record
        for record in records.values()
        if all(
            key in record
            for key in [
                "prompt_ms",
                "prompt_tokens",
                "prompt_tokens_per_second",
                "generation_ms",
                "generated_tokens",
                "generation_tokens_per_second",
                "total_ms",
                "context_tokens",
            ]
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--tokenizer-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    experiment = json.loads(args.runs.read_text(encoding="utf-8"))
    games = experiment.get("games", [])
    if len(games) != 1:
        raise SystemExit(f"Expected one preregistered structure result; found {len(games)}")
    game = games[0]
    turns = []
    for turn in game.get("turns", []):
        responses = turn.get("director_responses", {})
        transcript = turn.get("director_discussion_full", "")
        delivered_roles = re.findall(r"(?m)^(D[123]): ", transcript)
        messages = []
        for role in ["D1", "D2", "D3"]:
            response = responses.get(role)
            if not response:
                continue
            public_message = response.get("public_message", "")
            parsed = bool(public_message and public_message != "No message provided")
            present = role in delivered_roles
            messages.append(
                {
                    "role": role,
                    "parsed": parsed,
                    "delivered": present,
                    "message": public_message if parsed else "",
                    "message_token_count_with_role_label": tokenize(
                        f"{role}: {public_message}", args.tokenizer_url
                    ) if present else 0,
                    "has_square_bracket_wrapper": public_message.startswith("[") and (
                        "]" in public_message
                    ),
                    "leftover_format_tags": len(re.findall(r"</?(?:think|message)>", public_message, re.IGNORECASE)),
                }
            )

        progress = turn.get("progress_data", {}).get("metrics", {}).get("overall_progress")
        move = turn.get("builder_move_raw", turn.get("progress_data", {}).get("move", {}))
        turns.append(
            {
                "turn": turn.get("turn_number"),
                "generated_posts": len(responses),
                "parsed_messages": sum(message["parsed"] for message in messages),
                "builder_delivered_messages": sum(message["delivered"] for message in messages),
                "builder_transcript_token_count": tokenize(transcript, args.tokenizer_url) if transcript else 0,
                "progress": progress,
                "builder_followed_oracle": turn.get("builder_followed_oracle"),
                "selected_move": {
                    key: move.get(key)
                    for key in ["action", "block", "position", "layer", "span_to"]
                    if key in move
                },
                "messages": messages,
            }
        )

    calls = server_calls(args.server_log)
    result = {
        "study": "CRAFT single-slot local inference feasibility v0.21",
        "source_revision": "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430",
        "model": "Qwen3-8B-Q4_K_M",
        "model_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        "server": {"context": 4096, "parallel_slots": 1, "batch_size": 1024, "ubatch_size": 256},
        "structure": {
            "id": game.get("structure_id"),
            "complexity": game.get("complexity"),
            "completed": game.get("completed"),
            "turns_taken": game.get("turns_taken"),
            "final_progress": game.get("final_progress"),
            "stopping_reason": game.get("stopping_reason"),
        },
        "turns": turns,
        "message_format_summary": {
            "messages_with_square_bracket_wrappers": sum(
                message["has_square_bracket_wrapper"] for turn in turns for message in turn["messages"]
            ),
            "messages_with_leftover_think_or_message_tags": sum(
                message["leftover_format_tags"] > 0 for turn in turns for message in turn["messages"]
            ),
        },
        "inference_calls": calls,
        "inference_summary": {
            "completed_calls_in_server_log": len(calls),
            "prompt_tokens_total": sum(call["prompt_tokens"] for call in calls),
            "generated_tokens_total": sum(call["generated_tokens"] for call in calls),
            "service_time_seconds_total": round(sum(call["total_ms"] for call in calls) / 1000, 3),
            "mean_prompt_tokens_per_second": round(statistics.mean(call["prompt_tokens_per_second"] for call in calls), 2) if calls else None,
            "mean_generation_tokens_per_second": round(statistics.mean(call["generation_tokens_per_second"] for call in calls), 2) if calls else None,
        },
        "limits": [
            "One medium structure and two turns; this is a feasibility calibration, not a comparison.",
            "Wire-message token counts are parsed public transcript tokens with role labels; repeated history in model prompts is counted separately as inference input tokens.",
            "No task completion, generalization, or superiority inference is supported.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CRAFT v0.21 reduced-context local feasibility result",
        "",
        f"- Structure: `{game.get('structure_id')}` ({game.get('complexity')}); completed: `{game.get('completed')}`.",
        f"- Turns: {game.get('turns_taken')}; final progress: {game.get('final_progress'):.3f}; stopping: `{game.get('stopping_reason')}`.",
        f"- Model calls: {len(calls)}; prompt tokens: {result['inference_summary']['prompt_tokens_total']}; generated tokens: {result['inference_summary']['generated_tokens_total']}.",
        f"- Summed llama.cpp service time: {result['inference_summary']['service_time_seconds_total']} s; mean prompt/generation throughput: {result['inference_summary']['mean_prompt_tokens_per_second']}/{result['inference_summary']['mean_generation_tokens_per_second']} tokens/s.",
        f"- Parser residuals: {result['message_format_summary']['messages_with_square_bracket_wrappers']}/6 messages retained square-bracket wrappers; {result['message_format_summary']['messages_with_leftover_think_or_message_tags']}/6 retained a format tag.",
        "",
        "| Turn | Generated | Parsed | Builder received | Progress | Selected move | Wire tokens incl. role labels |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for turn in turns:
        lines.append(
            f"| {turn['turn']} | {turn['generated_posts']} | {turn['parsed_messages']} | {turn['builder_delivered_messages']} | "
            f"{turn['progress']:.3f} | `{turn['selected_move']}` | {turn['builder_transcript_token_count']} |"
        )
    lines.extend(["", "## Public messages", ""])
    for turn in turns:
        lines.append(f"### Turn {turn['turn']}")
        lines.append("")
        for message in turn["messages"]:
            lines.append(f"- **{message['role']}** (parsed={message['parsed']}, delivered={message['delivered']}, tokens={message['message_token_count_with_role_label']}, leftover format tags={message['leftover_format_tags']}): {message['message']}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "All generated, parsed, and Builder-delivered counts are kept separate. This run establishes only that the reduced-context local configuration completed the small CRAFT slice. Progress remained partial. It does not compare languages, estimate generalization, or establish a communication-efficiency advantage.",
            "",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
