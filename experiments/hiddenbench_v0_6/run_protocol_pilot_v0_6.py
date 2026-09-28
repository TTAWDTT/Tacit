"""Run the three frozen v0.6 protocol conditions for one local pilot task."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".cache/research/HiddenBench_ICML"
DATA = SOURCE / "src/hiddenbench/data/benchmark_short.json"
MODEL_PATH = ROOT / ".cache/models/Qwen3-14B-Q4_K_M.gguf"
MODEL = "Qwen3-14B-Q4_K_M"
MODEL_BYTES = 9_001_752_960
MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"
TASK_NAME = "evacuation_west_city"
SEED = 20260927
PROTOCOLS = ("natural_3", "exchange_decide", "reveal_all_3")
RUNS_ROOT = ROOT / ".cache/pilot_hiddenbench_v0_6"


def verify() -> dict:
    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REV:
        raise SystemExit(f"HiddenBench revision mismatch: {revision}")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit("HiddenBench benchmark checksum mismatch")
    if MODEL_PATH.stat().st_size != MODEL_BYTES or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("Qwen3-14B model artifact mismatch")
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        models = {item.get("id") for item in json.load(response).get("data", [])}
    if MODEL not in models:
        raise SystemExit(f"Expected local model {MODEL}; found {sorted(models)}")
    with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
        slots = json.load(response)
    if isinstance(slots, dict):
        slots = [slots]
    if len(slots) != 1 or slots[0].get("n_ctx") != 8192 or slots[0].get("is_processing"):
        raise SystemExit(f"Expected one idle 8192-token slot; got {slots}")
    return {"source_revision": revision, "slot": slots[0]}


def local_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
    env["OPENAI_API_KEY"] = "local-experiment"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def latest_server_task_id(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\| task (\d+)\s+\|", line)
        if match and "launch_slot_:" in line:
            latest = int(match.group(1))
    return latest


def run(*, prepare_only: bool = False, run_dir: Path | None = None, server_log: Path | None = None) -> None:
    os.environ.update(local_environment())
    sys.path.insert(0, str(SOURCE / "src"))
    from hiddenbench.benchmark import load_benchmark
    from hiddenbench.models import build_model_client
    from protocol_engine import run_scenario

    verified = verify()
    print(f"Pinned sources and idle local service verified: {verified}", flush=True)
    if prepare_only:
        return

    tasks = [task for task in load_benchmark(path=DATA) if task.name == TASK_NAME]
    if len(tasks) != 1:
        raise SystemExit(f"Expected exactly one {TASK_NAME} task; found {len(tasks)}")
    task = tasks[0]
    run_dir = run_dir or RUNS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"PILOT_RUN_DIR={run_dir}", flush=True)
    client = build_model_client(
        provider="openai-compatible",
        model=MODEL,
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="OPENAI_API_KEY",
        temperature=0.6,
    )
    for protocol in PROTOCOLS:
        output = run_dir / f"{protocol}.result.json"
        task_id_before = latest_server_task_id(server_log)
        result = run_scenario(task, client, "hidden", protocol, seed=SEED)
        task_id_after = latest_server_task_id(server_log)
        payload = {
            "metadata": {
                "benchmark": "HiddenBench",
                "profile": "hidden",
                "model": MODEL,
                "protocol": protocol,
                "rounds": 3,
                "seed": SEED,
                "task": TASK_NAME,
                "source_revision": SOURCE_REV,
                "server_task_id_before": task_id_before,
                "server_task_id_after": task_id_after,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "runs": [result],
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Completed {protocol}; private raw result saved locally at {output}", flush=True)
    print(f"Pilot complete: {run_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    run(prepare_only=args.prepare_only, run_dir=args.run_dir, server_log=args.server_log)
