"""Run preregistered HiddenBench v0.2 tasks with durable per-task outputs."""
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
RAW_ROOT = ROOT / ".cache/pilot_hiddenbench_v0_2"
MODEL = "Qwen3-8B-Q4_K_M"
MODEL_PATH = ROOT / ".cache/models/Qwen3-8B-Q4_K_M.gguf"
MODEL_SIZE = 5027783488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"
BASE_SEED = 20260927


def verify() -> None:
    rev = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if rev != SOURCE_REV:
        raise SystemExit(f"HiddenBench source revision mismatch: {rev}")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit("HiddenBench benchmark checksum mismatch")
    if MODEL_PATH.stat().st_size != MODEL_SIZE or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("Qwen3-8B model artifact mismatch")
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


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(SOURCE / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        "OPENAI_API_KEY": "local-experiment",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def run_profile(profile: str, run_dir: Path, tasks: list[dict]) -> None:
    profile_dir = run_dir / profile
    profile_dir.mkdir(parents=True)
    task_results = []
    for task_index, task in enumerate(tasks):
        task_name = task["name"]
        task_data = profile_dir / f"{task_name}.benchmark.json"
        task_data.write_text(json.dumps([task], ensure_ascii=False, indent=2), encoding="utf-8")
        output = profile_dir / f"{task_name}.result.json"
        log_path = profile_dir / f"{task_name}.runner.log"
        command = [
            sys.executable, "-m", "hiddenbench.cli", "eval",
            "--benchmark", str(task_data),
            "--provider", "openai-compatible",
            "--base-url", "http://127.0.0.1:8000/v1",
            "--api-key-env", "OPENAI_API_KEY",
            "--model", MODEL,
            "--temperature", "0.0",
            "--profile", profile,
            "--rounds", "15",
            "--duplications", "1",
            "--seed", str(BASE_SEED + task_index * 10_000),
            "--max-workers", "1",
            "--output", str(output),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=SOURCE, env=environment(), stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        print(f"{profile} {task_name}: exit={result.returncode}; raw={output}", flush=True)
        if result.returncode or not output.is_file():
            raise SystemExit(result.returncode or f"Missing task output for {task_name}")
        task_results.append(json.loads(output.read_text(encoding="utf-8")))
    merged = {
        "metadata": {
            "benchmark": "HiddenBench",
            "profile": profile,
            "model": MODEL,
            "rounds": 15,
            "duplications": 1,
            "base_seed": BASE_SEED,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "per_task_seed_rule": "base_seed + task_index * 10000",
        },
        "runs": [run for result in task_results for run in result["runs"]],
    }
    (profile_dir / "merged.result.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    verify()
    print("Pinned HiddenBench, benchmark, model, and idle 8192-context service verified.", flush=True)
    if args.prepare_only:
        return
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    expected = ["evacuation_west_city", "evacuation_north_hill", "evacuation_east_town"]
    tasks = [item for item in raw if item["name"] in expected]
    if [item["name"] for item in tasks] != expected:
        raise SystemExit("The pinned verification tasks are missing or out of order")
    run_dir = RAW_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    for profile in ("full", "hidden"):
        run_profile(profile, run_dir, tasks)
    print(f"Complete raw run: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
