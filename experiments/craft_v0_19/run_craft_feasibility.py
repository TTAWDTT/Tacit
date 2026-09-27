"""Run the frozen two-turn Qwen3-8B CRAFT feasibility calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache" / "research" / "CRAFT"
PATCHED = ROOT / ".cache" / "research" / "CRAFT-tacit-v0_19"
RAW = ROOT / ".cache" / "pilot_v0_19"
MODEL_PATH = ROOT / ".cache" / "models" / "Qwen3-8B-Q4_K_M.gguf"
UPSTREAM_REV = "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430"
MODEL = "Qwen3-8B-Q4_K_M"
MODEL_BYTES = 5027783488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"


def verify_model() -> None:
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise SystemExit(f"Expected exact Qwen3-8B artifact ({MODEL_BYTES} bytes) at {MODEL_PATH}")
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise SystemExit(f"Qwen3-8B SHA-256 mismatch: {digest.hexdigest()}")


def check_server() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Local llama.cpp server unavailable: {exc}") from exc
    model_ids = {item.get("id") for item in data.get("data", [])}
    if MODEL not in model_ids:
        raise SystemExit(f"Expected model alias {MODEL!r}; got {sorted(model_ids)}")


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_source() -> None:
    PATCHED.resolve().relative_to(ROOT.resolve())
    if git_revision(UPSTREAM) != UPSTREAM_REV:
        raise SystemExit(f"CRAFT source must be pinned at {UPSTREAM_REV}")
    if PATCHED.exists():
        shutil.rmtree(PATCHED)
    shutil.copytree(UPSTREAM, PATCHED, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    runner = PATCHED / "run_craft.py"
    source = runner.read_text(encoding="utf-8")
    old_schedule = 'director_order = random.choices(["D1", "D2", "D3"], k=3)'
    if source.count(old_schedule) != 1:
        raise SystemExit("Unexpected upstream scheduler; refusing to patch")
    source = source.replace(old_schedule, 'director_order = random.sample(["D1", "D2", "D3"], k=3)')
    seed_anchor = "    RUN           = args.run\n"
    if source.count(seed_anchor) != 1:
        raise SystemExit("Unexpected upstream seed location; refusing to patch")
    runner.write_text(source.replace(seed_anchor, seed_anchor + "    random.seed(RUN)\n"), encoding="utf-8")

    director = PATCHED / "agents" / "director_agent.py"
    source = director.read_text(encoding="utf-8")
    old = '        cleaned = re.sub(r\'\\[.*?\\]\', \'\', response_text, flags=re.DOTALL).strip()'
    new = '''        bracketed = re.findall(r"\\[([^\\[\\]]+)\\]", response_text, flags=re.DOTALL)
        bracketed = [
            part.strip()
            for part in bracketed
            if part.strip() and "natural human speech only" not in part.lower()
        ]
        if bracketed:
            cleaned = "\\n".join(bracketed)
        else:
            cleaned = re.sub(
                r"(?im)^\\s*\\[Natural human speech only[^\\]]*\\]\\s*$",
                "",
                response_text,
            ).strip()'''
    if source.count(old) != 1:
        raise SystemExit("Unexpected upstream Director parser; refusing to patch")
    director.write_text(source.replace(old, new), encoding="utf-8")
    (PATCHED / "tacit_adapter_manifest.json").write_text(
        json.dumps(
            {
                "upstream_revision": UPSTREAM_REV,
                "model_sha256": MODEL_SHA256,
                "patches": [
                    "seeded distinct-role scheduling",
                    "preserve bracketed Director prose except instruction echo",
                    "UTF-8 console I/O",
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    verify_model()
    check_server()
    prepare_source()
    print("Pinned CRAFT source, Qwen3-8B artifact, parser adapter, and endpoint verified.")
    if args.prepare_only:
        return

    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
    env["OPENAI_API_KEY"] = "local-experiment"
    env["PYTHONHASHSEED"] = "319"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / ".cache" / "python-packages") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "run_craft.py",
        "--mode", "api",
        "--director", MODEL,
        "--builder", MODEL,
        "--dataset", "data/structures_dataset_20.json",
        "--structures", "0",
        "--turns", "2",
        "--run", "319",
        "--oracle",
        "--oracle_n", "5",
        "--no_tools",
        "--output", str(RAW / "upstream_results"),
    ]
    log_path = RAW / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=PATCHED, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"Runner exit code: {proc.returncode}; log: {log_path}")
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
