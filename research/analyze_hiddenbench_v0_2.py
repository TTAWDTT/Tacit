"""Create a privacy-sanitized aggregate from the local HiddenBench v0.2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML/src"
DATA = SOURCE / "hiddenbench/data/benchmark_short.json"
MODEL = "Qwen3-8B-Q4_K_M"


def tokenize(text: str) -> int:
    payload = json.dumps({"content": text, "add_special": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/tokenize", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return len(json.load(response)["tokens"])


def service_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    prompt_matches = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    prompt_tokens = [int(tokens) for _, tokens in prompt_matches]
    generated = [int(tokens) for _, tokens in re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)]
    # Match only the total-time rows, not prompt/eval-time rows.
    total_ms = [float(ms) for ms, _ in re.findall(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens", text)]
    context_use = [(int(tokens), int(flag)) for tokens, flag in re.findall(r"n_tokens = (\d+), truncated = (\d+)", text)]
    truncated = sum(flag for _, flag in context_use)
    return {
        "server_requests": len(re.findall(r"processing task, is_child = 0", text)),
        "server_completions": len(re.findall(r"stop processing: n_tokens =", text)),
        "prompt_eval_tokens_reported": sum(prompt_tokens),
        "generated_tokens": sum(generated),
        "summed_service_ms": round(sum(total_ms), 3),
        "mean_service_ms_per_request": round(sum(total_ms) / len(total_ms), 3) if total_ms else None,
        "max_context_tokens_seen": max((tokens for tokens, _ in context_use), default=0),
        "truncated_completions": truncated,
        "prompt_eval_events": len(prompt_matches),
        "total_time_events": len(total_ms),
    }


def score_and_wire(profile: str, merged_path: Path, benchmark, scorer) -> dict:
    results = json.loads(merged_path.read_text(encoding="utf-8"))
    scores = scorer(results, benchmark)
    run_rows = []
    total_wire_tokens = 0
    total_wire_bytes = 0
    message_count = 0
    for run in results["runs"]:
        serialized = "".join(
            f"{message['agent']}: {message['response']}\n"
            for message in run["discussion_transcript"]
        )
        wire_tokens = tokenize(serialized)
        wire_bytes = len(serialized.encode("utf-8"))
        total_wire_tokens += wire_tokens
        total_wire_bytes += wire_bytes
        message_count += len(run["discussion_transcript"])
        per_task = next(row for row in scores["by_task"] if row["scenario"] == run["scenario"])
        run_rows.append({
            "task": run["scenario"],
            "seed": run["seed"],
            "messages": len(run["discussion_transcript"]),
            "discussion_wire_tokens": wire_tokens,
            "discussion_wire_utf8_bytes": wire_bytes,
            "initial_accuracy": per_task["pre_average_accuracy"],
            "final_accuracy": per_task["post_average_accuracy"],
            "initial_majority_correct": per_task["pre_majority_accuracy"],
            "final_majority_correct": per_task["post_majority_accuracy"],
        })
    return {
        "profile": profile,
        "official_hiddenbench_scores": scores,
        "per_task": run_rows,
        "discussion_transport": {
            "messages": message_count,
            "serialized_wire_tokens": total_wire_tokens,
            "serialized_wire_utf8_bytes": total_wire_bytes,
            "mean_tokens_per_message": round(total_wire_tokens / message_count, 3) if message_count else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.metrics import score_results

    benchmark = load_benchmark(path=DATA)
    profile_data = {
        profile: score_and_wire(profile, args.raw_dir / profile / "merged.result.json", benchmark, score_results)
        for profile in ("full", "hidden")
    }
    service = service_metrics(args.server_log)
    metrics = {
        "study": "HiddenBench local capability calibration",
        "version": "0.2",
        "model": MODEL,
        "source_revision": "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1",
        "design": {
            "tasks": 3,
            "agents_per_task": 4,
            "rounds": 15,
            "profiles": ["full", "hidden"],
            "planned_successful_completions": 408,
            "context_tokens": 8192,
        },
        "conditions": profile_data,
        "service": service,
        "capability_gate": {
            "threshold": 0.8,
            "full_profile_initial_average_accuracy": profile_data["full"]["official_hiddenbench_scores"]["average_accuracy"]["pre"],
            "passed": profile_data["full"]["official_hiddenbench_scores"]["average_accuracy"]["pre"] >= 0.8,
        },
        "interpretation": "Local capability calibration only. Three verification tasks, one seed schedule, and one model do not support protocol, generalization, scaling, or frontier claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "full_pre_average": metrics["conditions"]["full"]["official_hiddenbench_scores"]["average_accuracy"]["pre"],
        "hidden_pre_average": metrics["conditions"]["hidden"]["official_hiddenbench_scores"]["average_accuracy"]["pre"],
        "requests": service["server_requests"],
        "truncated": service["truncated_completions"],
    }, indent=2))


if __name__ == "__main__":
    main()
