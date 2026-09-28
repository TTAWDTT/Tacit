"""Run the preregistered four-agent, full-information HiddenBench screen."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML"
DATA = SOURCE / "src/hiddenbench/data/benchmark_short.json"
MODEL_DIR = ROOT / ".cache/models/Qwen3-1.7B"
MODEL = "Qwen3-1.7B"
SHARDS = {
    "model-00001-of-00002.safetensors": (
        3441185608,
        "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    ),
    "model-00002-of-00002.safetensors": (
        622329984,
        "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    ),
}
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"
TASK = "evacuation_west_city"
SEED = 20260927
RAW_ROOT = ROOT / ".cache/pilot_hiddenbench_v0_9"
MODEL_URL = "http://127.0.0.1:8000/v1/models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts() -> None:
    revision = __import__("subprocess").run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REV:
        raise SystemExit(f"HiddenBench source revision mismatch: {revision}")
    if sha256_file(DATA) != DATA_SHA256:
        raise SystemExit("HiddenBench benchmark checksum mismatch")
    for filename, (expected_size, expected_hash) in SHARDS.items():
        path = MODEL_DIR / filename
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise SystemExit(f"Qwen3-1.7B artifact mismatch: {filename}")


def verify_service() -> None:
    with urllib.request.urlopen(MODEL_URL, timeout=5) as response:
        models = json.load(response).get("data", [])
    if MODEL not in {item.get("id") for item in models}:
        raise SystemExit(f"Expected local model alias {MODEL}; service returned {models}")


def atomic_write_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()

    verify_artifacts()
    if args.artifacts_only:
        print("Pinned HiddenBench source and Qwen3-1.7B artifacts verified.", flush=True)
        return
    verify_service()

    os.environ["OPENAI_API_KEY"] = "local-experiment"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.path.insert(0, str(SOURCE / "src"))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.metrics import score_results
    from hiddenbench.models import OpenAICompatibleClient
    from hiddenbench.prompts import load_prompts, render_prompt
    from hiddenbench.simulator import Profile, instantiate_agents

    benchmark = load_benchmark(path=DATA)
    task = next((item for item in benchmark if item.name == TASK), None)
    if task is None:
        raise SystemExit(f"Pinned benchmark is missing {TASK}")
    client = OpenAICompatibleClient(
        model=MODEL,
        provider="openai-compatible",
        api_key="local-experiment",
        base_url="http://127.0.0.1:8000/v1",
        temperature=0.0,
        max_retries=1,
        timeout=90.0,
    )
    prompts = load_prompts()
    agents, assignments = instantiate_agents(
        task,
        client,
        Profile.FULL,
        prompts,
        random.Random(SEED),
        extra_prompt="",
        special_agent_ratio=1.0,
    )
    if len(agents) != 4:
        raise SystemExit(f"Expected four agents; benchmark produced {len(agents)}")

    vote_prompt = render_prompt(
        prompts.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    started = datetime.now(timezone.utc)
    raw_dir = RAW_ROOT / started.strftime("%Y%m%dT%H%M%SZ")
    raw_dir.mkdir(parents=True, exist_ok=False)
    raw_path = raw_dir / "full_profile.raw.json"
    usage_path = Path(os.environ["TLU_USAGE_LOG"])
    record = {
        "task_id": task.id,
        "scenario": task.name,
        "correct_answer": task.correct_answer,
        "profile": "full",
        "model": MODEL,
        "seed": SEED,
        "fact_assignments": assignments,
        "initial_votes": [],
    }
    atomic_write_json(raw_path, record)

    votes = []
    for agent in agents:
        response = agent.vote(vote_prompt, task.possible_answers, retries=1)
        vote = {"agent": agent.name, **response}
        votes.append(vote)
        record["initial_votes"] = votes
        atomic_write_json(raw_path, record)

    if len(votes) != 4:
        raise SystemExit(f"Expected four initial votes, got {len(votes)}")
    usage_records = [
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(usage_records) != len(votes):
        raise SystemExit(
            f"Expected one server usage record per vote ({len(votes)}); got {len(usage_records)}"
        )
    if any(item.get("finish_reason") != "stop" for item in usage_records):
        raise SystemExit("At least one response hit the output cap; screen is incomplete")

    official = score_results(
        {
            "metadata": {"profile": "full", "model": MODEL},
            "runs": [
                {
                    "task_id": task.id,
                    "scenario": task.name,
                    "profile": "full",
                    "seed": SEED,
                    "initial_votes": votes,
                    "final_votes": votes,
                    "discussion_transcript": [],
                }
            ],
        },
        benchmark,
    )
    correct_votes = sum(vote.get("vote") == task.correct_answer for vote in votes)
    result = {
        "study": "HiddenBench Qwen3-1.7B full-information capability screen",
        "version": "0.9",
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
        "model_requests": len(usage_records),
        "prompt_tokens": sum(item["prompt_tokens"] for item in usage_records),
        "completion_tokens": sum(item["completion_tokens"] for item in usage_records),
        "model_generation_seconds": round(
            sum(item["generation_seconds"] for item in usage_records), 3
        ),
        "usage_log_local_path": str(usage_path.relative_to(ROOT)),
        "raw_result_local_path": str(raw_path.relative_to(ROOT)),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Single-task, full-information capability/resource screen only; no communication or protocol claim.",
    }
    atomic_write_json(raw_dir / "sanitized_result.local.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
