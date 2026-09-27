"""Sanitize CRAFT v0.24 traces and summarize the communication-necessity ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import urllib.request

from analyze_craft_v0_23 import llama_calls, tokenize

EXPECTED = [
    ("structure_001", "medium"),
    ("structure_008", "simple"),
    ("structure_003", "complex"),
]


def load_games(root: Path, condition: str) -> dict[str, dict]:
    games = {}
    for structure_id, level in EXPECTED:
        found = list((root / condition).rglob(f"craft_{structure_id}_323.json"))
        if len(found) != 1:
            raise SystemExit(f"{condition}: expected one result for {structure_id}, found {len(found)}")
        game = json.loads(found[0].read_text(encoding="utf-8"))["games"][0]
        if game.get("complexity") != level:
            raise SystemExit(f"Unexpected complexity for {structure_id}: {game.get('complexity')}")
        games[structure_id] = game
    return games


def summarize_condition(root: Path, condition: str, *, tokenizer_url: str, random_policy: bool) -> dict:
    games = load_games(root, condition)
    structures = []
    for structure_id, level in EXPECTED:
        game = games[structure_id]
        turns = []
        for turn in game.get("turns", []):
            discussion = turn.get("director_discussion_full", "")
            delivered_roles = re.findall(r"(?m)^(D[123]): ", discussion)
            responses = turn.get("director_responses", {})
            successful = [
                role for role, response in responses.items()
                if response.get("raw_response") and response.get("public_message") not in ("", "No message provided")
            ]
            attempted = turn.get("move_attempted", {}) or {}
            oracle_moves = turn.get("oracle_moves", []) or []
            progress = turn.get("progress_data", {}).get("metrics", {}).get("overall_progress")
            move = {key: attempted.get(key) for key in ("action", "block", "position", "layer", "span_to") if key in attempted}
            turns.append({
                "turn": turn.get("turn_number"),
                "oracle_candidates": len(oracle_moves),
                "message_slots_successful": len(successful),
                "messages_delivered": len(delivered_roles),
                "wire_transcript_tokens": tokenize(discussion, tokenizer_url) if discussion else 0,
                "move_executed": turn.get("move_executed"),
                "followed_oracle": turn.get("builder_followed_oracle"),
                "action": move,
                "progress": progress,
                "confirmation_generated": bool(turn.get("builder_confirmation")),
            })
        initial = game.get("turns", [{}])[0].get("structure_before", {}) if turns else {}
        structures.append({
            "id": structure_id,
            "complexity": level,
            "initial_blocks": sum(len(stack) for stack in initial.values()),
            "partial_completion_category": game.get("partialCompletionCategory"),
            "completed": game.get("completed"),
            "turns_taken": game.get("turns_taken"),
            "final_progress": game.get("final_progress"),
            "successful_moves": sum(t["move_executed"] is True for t in turns),
            "failed_or_clarified_moves": sum(t["move_executed"] is False for t in turns),
            "oracle_followed_moves": sum(t["followed_oracle"] is True for t in turns),
            "turns": turns,
        })
    flat = [turn for structure in structures for turn in structure["turns"]]
    successful = sum(turn["message_slots_successful"] for turn in flat)
    delivered = sum(turn["messages_delivered"] for turn in flat)
    return {
        "structures": structures,
        "descriptive_means": {
            "final_progress": round(statistics.mean(s["final_progress"] for s in structures), 6),
            "completed_structures": sum(bool(s["completed"]) for s in structures),
            "model_public_messages_successful": successful,
            "builder_delivered_messages": delivered,
            "wire_tokens_per_turn_mean": round(statistics.mean(t["wire_transcript_tokens"] for t in flat), 2),
            "oracle_followed_moves": sum(t["followed_oracle"] is True for t in flat),
            "successful_moves": sum(t["move_executed"] is True for t in flat),
            "failed_moves": sum(t["move_executed"] is False for t in flat),
            "oracle_empty_turns": sum(t["oracle_candidates"] == 0 for t in flat),
            "turns_with_any_model_confirmation": sum(t["confirmation_generated"] for t in flat),
        },
    }


def call_summary(calls: list[dict]) -> dict:
    return {
        "completed_calls": len(calls),
        "prompt_eval_tokens": sum(call["prompt_tokens"] for call in calls),
        "generated_tokens": sum(call["generated_tokens"] for call in calls),
        "summed_service_seconds": round(sum(call["total_ms"] for call in calls) / 1000, 3),
        "mean_prompt_eval_tokens_per_second": round(statistics.mean(c["prompt_tps"] for c in calls), 2) if calls else None,
        "mean_generation_tokens_per_second": round(statistics.mean(c["generation_tps"] for c in calls), 2) if calls else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--prior-completed-calls", type=int, default=154,
                        help="v0.21/v0.22/v0.23 calls preceding v0.24 in the shared llama.cpp log")
    parser.add_argument("--initial-deviation-calls", type=int, default=14,
                        help="calls caused by the first random-policy implementation's empty-list fallthrough")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--tokenizer-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    calls = llama_calls(args.server_log, args.prior_completed_calls)
    deviation = args.initial_deviation_calls
    expected = {"natural_language_team": 240, "zero_message_llm_builder": 60}
    if len(calls) != sum(expected.values()) + deviation:
        raise SystemExit(f"Expected {sum(expected.values()) + deviation} post-v0.23 calls, found {len(calls)}")
    offset = 0
    call_conditions = {}
    for condition, count in expected.items():
        call_conditions[condition] = call_summary(calls[offset:offset + count])
        offset += count
    call_conditions["zero_message_random_oracle_policy"] = call_summary([])
    deviation_calls = call_summary(calls[offset:])

    conditions = {}
    for name, random_policy in [
        ("natural_language_team", False),
        ("zero_message_llm_builder", False),
        ("zero_message_random_oracle_policy", True),
    ]:
        conditions[name] = summarize_condition(
            args.runs_dir, name, tokenizer_url=args.tokenizer_url, random_policy=random_policy
        )
    result = {
        "study": "CRAFT v0.24 oracle communication-necessity ablation",
        "source_revision": "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430",
        "model": "Qwen3-8B-Q4_K_M",
        "model_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        "design": {"structures": ["structure_001", "structure_008", "structure_003"],
                   "run_seed": 323, "turn_limit": 20, "oracle_candidates": 5,
                   "history_window_lines": 16, "server_context": 4096},
        "conditions": conditions,
        "inference": call_conditions,
        "implementation_deviation": {
            "first_random_policy_run": "empty oracle list fell through to Builder inference; excluded from primary control",
            "deviation_model_calls": deviation_calls,
            "corrected_random_policy_rerun": "no model calls; empty oracle list returns clarification",
        },
        "interpretation": [
            "At least one zero-message policy completed structure_008 under the shared oracle; strict communication necessity is false for that instance in this oracle-assisted setup.",
            "The oracle uses target-grounded candidates, so this does not imply message-free agents can generally build from private observations without oracle assistance.",
            "The one-seed, three-structure, one-model result is descriptive and does not rank natural language against other protocols.",
        ],
        "limits": [
            "One run seed and three structures; completion is thresholded by the upstream game scorer.",
            "Zero-message policies still receive privileged target-grounded oracle actions.",
            "The deterministic-policy control's initial implementation deviation is retained and reported separately.",
            "Prompt-eval token counts are llama.cpp measured prompt-evaluation tokens and can exclude KV-cache-reused prefix tokens; they are not a direct client-side billing-token count.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CRAFT v0.24 oracle communication-necessity ablation",
        "",
        "Single-seed benchmark-validity audit; not a protocol comparison.",
        "",
        "| Structure | Condition | Turns | Progress | Complete | Successful moves | Failed moves | Oracle-followed |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    labels = {"natural_language_team": "Natural-language team", "zero_message_llm_builder": "Zero-message LLM Builder", "zero_message_random_oracle_policy": "Zero-message random oracle"}
    for condition, details in conditions.items():
        for item in details["structures"]:
            lines.append(
                f"| {item['id']} ({item['complexity']}) | {labels[condition]} | {item['turns_taken']} | "
                f"{item['final_progress']:.3f} | {item['completed']} | {item['successful_moves']} | "
                f"{item['failed_or_clarified_moves']} | {item['oracle_followed_moves']} |"
            )
    lines.extend([
        "",
        "## Main observation",
        "",
        "Both zero-message conditions completed the simple structure in 15 turns (upstream completion threshold: 0.95); the natural-language team did not complete any of the three structures within 20 turns. The random-policy trace chose a listed oracle action on all 15 turns and invoked no model. This is a counterexample to strict communication necessity for this structure under the oracle-assisted setup, not evidence that natural language is generally worse.",
        "",
        "The oracle candidate list is target-grounded and the Builder never receives the Directors' private views directly. Therefore the result tests this benchmark configuration with its privileged action oracle. It does not answer whether ordinary LLM agents can collaborate without communication when the oracle is removed.",
        "",
        "## Inference and implementation audit",
        "",
        f"The natural-language condition used {call_conditions['natural_language_team']['completed_calls']} completed server calls; the zero-message LLM Builder used {call_conditions['zero_message_llm_builder']['completed_calls']}; the corrected random-oracle condition used 0. The corrected control's empty-candidate action is a clarification and does not call the model.",
        f"The first random-policy implementation made {deviation_calls['completed_calls']} unintended model calls after the candidate list became empty. Those data are excluded from the primary result and described in [the implementation audit](../experiments/craft_v0_24/IMPLEMENTATION_AUDIT.md). The corrected full three-structure control is the reported result.",
        "",
        "This is one seed, three structures, and one local model. It establishes benchmark eligibility concerns only; it does not establish protocol superiority, generalization, or an efficiency frontier.",
    ])
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_report}")


if __name__ == "__main__":
    main()
