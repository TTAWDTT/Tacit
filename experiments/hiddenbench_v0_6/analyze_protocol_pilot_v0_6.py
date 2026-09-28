"""Create sanitized scores and per-condition cost metrics for v0.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML/src"
DATA = SOURCE / "hiddenbench/data/benchmark_short.json"
PROTOCOLS = ("natural_3", "exchange_decide", "reveal_all_3")


def tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return len(json.load(response)["tokens"])


def _task_id(line: str) -> int | None:
    match = re.search(r"\| task (\d+)\s+\|", line)
    return int(match.group(1)) if match else None


def server_metrics(path: Path, before: int | None, after: int | None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Server log not found: {path}")
    selected = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        task_id = _task_id(line)
        if task_id is None:
            continue
        if before is not None and task_id <= before:
            continue
        if after is not None and task_id > after:
            continue
        selected.append(line)
    text = "\n".join(selected)
    prompt_events = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    generated_events = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    total_events = re.findall(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens", text)
    contexts = [(int(size), int(truncated)) for size, truncated in re.findall(
        r"n_tokens = (\d+), truncated = (\d+)", text
    )]
    total_ms = sum(float(milliseconds) for milliseconds, _ in total_events)
    launches = len(re.findall(r"launch_slot_:.*processing task, is_child = 0", text))
    completions = len(re.findall(r"stop processing: n_tokens =", text))
    return {
        "server_task_id_before_exclusive": before,
        "server_task_id_after_inclusive": after,
        "server_requests": launches,
        "server_completions": completions,
        "prompt_eval_tokens_reported": sum(int(tokens) for _, tokens in prompt_events),
        "generated_tokens": sum(int(tokens) for _, tokens in generated_events),
        "summed_service_ms": round(total_ms, 3),
        "mean_service_ms_per_request": round(total_ms / len(total_events), 3) if total_events else None,
        "max_context_tokens_seen": max((size for size, _ in contexts), default=0),
        "truncated_completions": sum(truncated for _, truncated in contexts),
    }


def resource_metrics(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not samples:
        return {"sample_count": 0}
    fields = (
        "server_cpu_cores",
        "server_cpu_percent_system",
        "server_working_set_mib",
        "host_cpu_percent",
        "gpu_util_percent",
    )
    summary = {"sample_count": len(samples)}
    for field in fields:
        values = [float(sample[field]) for sample in samples if sample.get(field) is not None]
        if values:
            summary[field] = {
                "mean": round(sum(values) / len(values), 3),
                "max": round(max(values), 3),
            }
    return summary


def analyze(raw_dir: Path, server_log: Path, output: Path, tokenize_fn=tokenize,
            resource_log: Path | None = None) -> dict:
    sys.path.insert(0, str(SOURCE))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.metrics import score_results

    benchmark = load_benchmark(path=DATA)
    task = "evacuation_west_city"
    rows = []
    for protocol in PROTOCOLS:
        raw_path = raw_dir / f"{protocol}.result.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing completed condition output: {raw_path}")
        result_set = json.loads(raw_path.read_text(encoding="utf-8"))
        if result_set.get("metadata", {}).get("protocol") != protocol:
            raise ValueError(f"Protocol metadata mismatch in {raw_path}")
        run = result_set["runs"][0]
        if run.get("scenario") != task or run.get("profile") != "hidden":
            raise ValueError(f"Unexpected task/profile in {raw_path}")
        score = score_results(result_set, benchmark)["by_task"][0]
        transcript = "".join(
            f"Round {item['round']} {item['agent']}: {item['response']}\n"
            for item in run["discussion_transcript"]
        )
        wire_bytes = len(transcript.encode("utf-8"))
        wire_tokens = tokenize_fn(transcript)
        metadata = result_set["metadata"]
        rows.append({
            "protocol": protocol,
            "task": task,
            "seed": run["seed"],
            "messages": len(run["discussion_transcript"]),
            "discussion_payload_tokens": wire_tokens,
            "discussion_payload_utf8_bytes": wire_bytes,
            "initial_individual_accuracy": score["pre_average_accuracy"],
            "final_individual_accuracy": score["post_average_accuracy"],
            "initial_majority_accuracy": score["pre_majority_accuracy"],
            "final_majority_accuracy": score["post_majority_accuracy"],
            "service": server_metrics(
                server_log,
                metadata.get("server_task_id_before"),
                metadata.get("server_task_id_after"),
            ),
        })
    report = {
        "study": "HiddenBench phase-policy feasibility pilot",
        "version": "0.6",
        "model": "Qwen3-14B-Q4_K_M",
        "benchmark_revision": "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1",
        "design": {"task": task, "profile": "hidden", "agents": 4, "rounds": 3, "seed": 20260927},
        "conditions": rows,
        "host_resource_samples": resource_metrics(resource_log),
        "interpretation": (
            "One task, one seed, and one model per condition are feasibility observations only. "
            "Reveal-All is an information-policy diagnostic, not a bandwidth-matched encoding comparison. "
            "Serialized discussion payload tokens exclude system prompts, vote prompts, and model-side inference compute."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--resource-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.raw_dir, args.server_log, args.output, resource_log=args.resource_log)
    print(json.dumps({"output": str(args.output), "conditions": report["conditions"]}, indent=2))


if __name__ == "__main__":
    main()
