"""Run a four-vote Qwen3-8B full-information capability screen."""
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
MODEL_PATH = ROOT / ".cache/models/Qwen3-8B-Q4_K_M.gguf"
MODEL = "Qwen3-8B-Q4_K_M"
MODEL_SIZE = 5027783488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"
TASK = "evacuation_west_city"
SEED = 20260927
RAW_ROOT = ROOT / ".cache/pilot_hiddenbench_v0_8"


def verify_artifacts() -> None:
    revision = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REV:
        raise SystemExit(f"HiddenBench source revision mismatch: {revision}")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit("HiddenBench benchmark checksum mismatch")
    if MODEL_PATH.stat().st_size != MODEL_SIZE or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("Qwen3-8B model artifact mismatch")


def verify_service() -> None:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        aliases = {item.get("id") for item in json.load(response).get("data", [])}
    if MODEL not in aliases:
        raise SystemExit(f"Expected model alias {MODEL}; found {sorted(aliases)}")
    with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
        slots = json.load(response)
    if isinstance(slots, dict):
        slots = [slots]
    if len(slots) != 1 or slots[0].get("n_ctx") != 8192 or slots[0].get("is_processing"):
        raise SystemExit(f"Expected one idle 8192-token context slot; got {slots}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()
    verify_artifacts()
    if args.artifacts_only:
        print("Pinned HiddenBench source, dataset, and Qwen3-8B model artifacts verified.", flush=True)
        return
    verify_service()
    print("Pinned artifacts and idle Qwen3-8B service verified.", flush=True)

    os.environ["OPENAI_API_KEY"] = "local-experiment"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.path.insert(0, str(SOURCE / "src"))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.metrics import score_results
    from hiddenbench.models import build_model_client
    from hiddenbench.prompts import load_prompts
    from hiddenbench.simulator import Profile, collect_initial_votes

    benchmark = load_benchmark(path=DATA)
    task = next((item for item in benchmark if item.name == TASK), None)
    if task is None:
        raise SystemExit(f"Pinned benchmark is missing {TASK}")
    client = build_model_client(
        provider="openai-compatible", model=MODEL,
        api_key_env="OPENAI_API_KEY", base_url="http://127.0.0.1:8000/v1",
        temperature=0.6,
    )
    started = datetime.now(timezone.utc)
    votes, assignments = collect_initial_votes(
        task, client, Profile.FULL, load_prompts(), seed=SEED,
        extra_prompt="", special_agent_ratio=1.0,
    )
    if len(votes) != 4:
        raise SystemExit(f"Expected 4 initial votes, got {len(votes)}; screen incomplete")
    raw_dir = RAW_ROOT / started.strftime("%Y%m%dT%H%M%SZ")
    raw_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "task_id": task.id,
        "scenario": task.name,
        "correct_answer": task.correct_answer,
        "profile": "full",
        "model": MODEL,
        "seed": SEED,
        "fact_assignments": assignments,
        "initial_votes": votes,
    }
    raw_path = raw_dir / "full_profile.raw.json"
    raw_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    official = score_results({"metadata": {"profile": "full", "model": MODEL}, "runs": [{
        "task_id": task.id,
        "scenario": task.name,
        "profile": "full",
        "seed": SEED,
        "initial_votes": votes,
        "final_votes": votes,
        "discussion_transcript": [],
    }]}, benchmark)
    correct_votes = sum(vote.get("vote") == task.correct_answer for vote in votes)
    result = {
        "study": "HiddenBench Qwen3-8B partial-GPU capability screen",
        "version": "0.8",
        "status": "complete",
        "task": TASK,
        "profile": "full",
        "seed": SEED,
        "model": MODEL,
        "correct_votes": correct_votes,
        "votes": len(votes),
        "initial_individual_accuracy": correct_votes / len(votes),
        "eligibility_threshold": 1.0,
        "passed": correct_votes == 4,
        "official_score": official["by_task"][0],
        "raw_result_local_path": str(raw_path.relative_to(ROOT)),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Single-task capability screen only; it cannot establish protocol value or superiority.",
    }
    (raw_dir / "sanitized_result.local.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
