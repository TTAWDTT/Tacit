"""Run the preregistered HiddenBench v0.1 local capability calibration."""
from __future__ import annotations

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
RAW = ROOT / ".cache/pilot_hiddenbench_v0_1"
MODEL = "Qwen3-8B-Q4_K_M"
MODEL_PATH = ROOT / ".cache/models/Qwen3-8B-Q4_K_M.gguf"
MODEL_SIZE = 5027783488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SOURCE_REV = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
DATA_SHA256 = "8684c1c8b02da49bb3a959c9ba352b3eb682c02653a3c508c47b8ed54f7a5df3"


def verify() -> None:
    if not SOURCE.is_dir() or not DATA.is_file():
        raise SystemExit("Pinned HiddenBench source/data missing from project cache")
    rev = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if rev != SOURCE_REV:
        raise SystemExit(f"HiddenBench source revision mismatch: {rev}")
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if data_sha != DATA_SHA256:
        raise SystemExit(f"HiddenBench benchmark checksum mismatch: {data_sha}")
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL_SIZE:
        raise SystemExit("Qwen3-8B model size/path mismatch")
    model_sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_sha != MODEL_SHA256:
        raise SystemExit(f"Qwen3-8B model checksum mismatch: {model_sha}")
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        model_data = json.load(response)
    aliases = {item.get("id") for item in model_data.get("data", [])}
    if MODEL not in aliases:
        raise SystemExit(f"Expected model alias {MODEL}, found {sorted(aliases)}")
    with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
        slots = json.load(response)
    if isinstance(slots, dict):
        slots = [slots]
    if len(slots) != 1 or slots[0].get("n_ctx") != 4096 or slots[0].get("is_processing"):
        raise SystemExit(f"Expected one idle 4096-context slot, got {slots}")
    if not (SOURCE / "LICENSE").is_file():
        raise SystemExit("HiddenBench MIT license is missing")


def run_profile(profile: str) -> None:
    output = RAW / f"qwen3_8b_{profile}.json"
    log_path = RAW / f"qwen3_8b_{profile}.runner.log"
    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(SOURCE / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        "OPENAI_API_KEY": "local-experiment",
        "PYTHONIOENCODING": "utf-8",
    })
    command = [
        sys.executable, "-m", "hiddenbench.cli", "eval",
        "--benchmark", str(DATA),
        "--provider", "openai-compatible",
        "--base-url", "http://127.0.0.1:8000/v1",
        "--api-key-env", "OPENAI_API_KEY",
        "--model", MODEL,
        "--temperature", "0.0",
        "--profile", profile,
        "--rounds", "15",
        "--duplications", "1",
        "--seed", "20260927",
        "--max-workers", "1",
        "--output", str(output),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"{profile}: exit={result.returncode}; raw={output}; log={log_path}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    if not output.is_file():
        raise SystemExit(f"HiddenBench did not write {profile} result")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    verify()
    print("Pinned HiddenBench, benchmark, model, and local service verified.", flush=True)
    if args.prepare_only:
        return
    for profile in ("full", "hidden"):
        run_profile(profile)


if __name__ == "__main__":
    main()
