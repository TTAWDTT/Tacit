"""Create sanitized metrics for the Qwen3-14B HiddenBench baseline."""
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
MODEL = "Qwen3-14B-Q4_K_M"


def tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode("utf-8")
    request = urllib.request.Request("http://127.0.0.1:8000/tokenize", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return len(json.load(response)["tokens"])


def service_metrics(path: Path, after_task_id: int) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [
        line for line in lines
        if (match := re.search(r"\| task (\d+)\s+\|", line)) and int(match.group(1)) > after_task_id
    ]
    text = "\n".join(lines)
    prompt = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    generated = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    totals = re.findall(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    used = [(int(n), int(trunc)) for n, trunc in re.findall(r"n_tokens = (\d+), truncated = (\d+)", text)]
    latency = sum(float(ms) for ms, _ in totals)
    return {
        "server_requests": len(re.findall(r"processing task, is_child = 0", text)),
        "server_completions": len(re.findall(r"stop processing: n_tokens =", text)),
        "prompt_eval_tokens_reported": sum(int(tokens) for _, tokens in prompt),
        "generated_tokens": sum(int(tokens) for _, tokens in generated),
        "summed_service_ms": round(latency, 3),
        "mean_service_ms_per_request": round(latency / len(totals), 3) if totals else None,
        "max_context_tokens_seen": max((n for n, _ in used), default=0),
        "truncated_completions": sum(flag for _, flag in used),
        "prompt_eval_events": len(prompt),
        "total_time_events": len(totals),
    }


def condition(profile: str, raw_dir: Path, benchmark, score_results) -> dict:
    results = json.loads((raw_dir / profile / "merged.result.json").read_text(encoding="utf-8"))
    scores = score_results(results, benchmark)
    rows = []
    total_wire_tokens = total_wire_bytes = message_count = 0
    for run in results["runs"]:
        transcript = "".join(f"{m['agent']}: {m['response']}\n" for m in run["discussion_transcript"])
        wire_tokens = tokenize(transcript)
        wire_bytes = len(transcript.encode("utf-8"))
        total_wire_tokens += wire_tokens
        total_wire_bytes += wire_bytes
        message_count += len(run["discussion_transcript"])
        task_score = next(row for row in scores["by_task"] if row["scenario"] == run["scenario"])
        rows.append({
            "task": run["scenario"],
            "seed": run["seed"],
            "messages": len(run["discussion_transcript"]),
            "discussion_wire_tokens": wire_tokens,
            "discussion_wire_utf8_bytes": wire_bytes,
            "initial_accuracy": task_score["pre_average_accuracy"],
            "final_accuracy": task_score["post_average_accuracy"],
            "initial_majority_correct": task_score["pre_majority_accuracy"],
            "final_majority_correct": task_score["post_majority_accuracy"],
        })
    return {
        "profile": profile,
        "official_hiddenbench_scores": scores,
        "per_task": rows,
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
    parser.add_argument("--after-server-task-id", required=True, type=int,
                        help="Ignore earlier completions from the same persistent llama-server process.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.metrics import score_results

    benchmark = load_benchmark(path=DATA)
    conditions = {profile: condition(profile, args.raw_dir, benchmark, score_results)
                  for profile in ("full", "hidden")}
    service = service_metrics(args.server_log, args.after_server_task_id)
    report = {
        "study": "HiddenBench Qwen3-14B natural-discussion baseline",
        "version": "0.5",
        "model": MODEL,
        "benchmark_revision": "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1",
        "design": {
            "tasks": 3, "agents_per_task": 4, "rounds": 15,
            "profiles": ["full", "hidden"], "planned_successful_completions": 408,
            "server_log_inference_start_task_id_exclusive": args.after_server_task_id,
        },
        "inference": {
            "reasoning": "on", "reasoning_budget_tokens": 1024,
            "temperature": 0.6, "top_k": 20, "top_p": 0.95,
            "min_p": 0.0, "presence_penalty": 1.5,
            "gpu_layers": 24, "context_tokens": 8192,
        },
        "conditions": conditions,
        "service": service,
        "interpretation": "One model, three related tasks, and one run per profile are a local baseline, not protocol superiority or generalization evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "full_final_average": conditions["full"]["official_hiddenbench_scores"]["average_accuracy"]["post"],
        "hidden_final_average": conditions["hidden"]["official_hiddenbench_scores"]["average_accuracy"]["post"],
        "server_requests": service["server_requests"],
        "generated_tokens": service["generated_tokens"],
        "truncated": service["truncated_completions"],
    }, indent=2))


if __name__ == "__main__":
    main()
