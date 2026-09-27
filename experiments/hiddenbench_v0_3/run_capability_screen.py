"""Screen Qwen3-14B initial-vote capability using the pinned HiddenBench API."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML"
DATA = SOURCE / "src/hiddenbench/data/benchmark_short.json"
MODEL_PATH = ROOT / ".cache/models/Qwen3-14B-Q4_K_M.gguf"
MODEL = "Qwen3-14B-Q4_K_M"
MODEL_SIZE = 9001752960
MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"
BASE_SEED = 20260927
RAW_ROOT = ROOT / ".cache/pilot_hiddenbench_v0_3"


def verify() -> None:
    rev = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if rev != SOURCE_REV:
        raise SystemExit(f"HiddenBench source revision mismatch: {rev}")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit("HiddenBench benchmark checksum mismatch")
    if MODEL_PATH.stat().st_size != MODEL_SIZE or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("Qwen3-14B model artifact mismatch")
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        aliases = {item.get("id") for item in json.load(response).get("data", [])}
    if MODEL not in aliases:
        raise SystemExit(f"Expected model alias {MODEL}, found {sorted(aliases)}")
    with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
        slots = json.load(response)
    if isinstance(slots, dict):
        slots = [slots]
    if len(slots) != 1 or slots[0].get("n_ctx") != 8192 or slots[0].get("is_processing"):
        raise SystemExit(f"Expected one idle 8192-context slot, got {slots}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    verify()
    print("Pinned HiddenBench and Qwen3-14B service verified.", flush=True)
    if args.prepare_only:
        return

    sys.path.insert(0, str(SOURCE / "src"))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.models import build_model_client
    from hiddenbench.prompts import load_prompts
    from hiddenbench.simulator import Profile, collect_initial_votes

    os.environ["OPENAI_API_KEY"] = "local-experiment"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    tasks = load_benchmark(path=DATA)
    names = ["evacuation_west_city", "evacuation_north_hill", "evacuation_east_town"]
    tasks = [task for task in tasks if task.name in names]
    if [task.name for task in tasks] != names:
        raise SystemExit("Pinned verification tasks are missing or out of order")
    run_dir = RAW_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    client = build_model_client(
        provider="openai-compatible", model=MODEL,
        api_key_env="OPENAI_API_KEY", base_url="http://127.0.0.1:8000/v1",
        temperature=0.0,
    )
    prompts = load_prompts()
    records = []
    for task_index, task in enumerate(tasks):
        seed = BASE_SEED + task_index * 10_000
        votes, assignments = collect_initial_votes(
            task, client, Profile.FULL, prompts, seed=seed,
            extra_prompt="", special_agent_ratio=1.0,
        )
        record = {
            "task_id": task.id,
            "scenario": task.name,
            "correct_answer": task.correct_answer,
            "profile": "full",
            "model": MODEL,
            "seed": seed,
            "fact_assignments": assignments,
            "initial_votes": votes,
        }
        task_path = run_dir / f"{task.name}.raw.json"
        task_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(record)
        print(f"{task.name}: saved four full-profile initial votes to local cache.", flush=True)

    summary = []
    correct = 0
    total = 0
    for record in records:
        hits = sum(vote["vote"] == record["correct_answer"] for vote in record["initial_votes"])
        correct += hits
        total += len(record["initial_votes"])
        summary.append({
            "task": record["scenario"],
            "correct_votes": hits,
            "votes": len(record["initial_votes"]),
            "accuracy": hits / len(record["initial_votes"]),
        })
    public = {
        "study": "HiddenBench Qwen3-14B full-information capability screen",
        "version": "0.3",
        "model": MODEL,
        "tasks": summary,
        "correct_votes": correct,
        "votes": total,
        "mean_individual_accuracy": correct / total,
        "eligibility_threshold": 0.8,
        "passed": correct / total >= 0.8,
        "raw_run_directory": str(run_dir),
    }
    (run_dir / "aggregate.local.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(json.dumps({key: public[key] for key in ("correct_votes", "votes", "mean_individual_accuracy", "passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
