"""Run the frozen, non-comparative CRAFT feasibility calibration."""

from __future__ import annotations

import argparse
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
PATCHED = ROOT / ".cache" / "research" / "CRAFT-tacit-v0_17"
RAW = ROOT / ".cache" / "pilot_v0_17"
UPSTREAM_REV = "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430"
MODEL = "Qwen3-14B-Q4_K_M"


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def check_server() -> None:
    request = urllib.request.Request("http://127.0.0.1:8000/v1/models")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Local llama.cpp server unavailable: {exc}") from exc
    model_ids = {item.get("id") for item in data.get("data", [])}
    if MODEL not in model_ids:
        raise SystemExit(f"Expected model alias {MODEL!r}; server returned {sorted(model_ids)}")


def prepare_patched_source() -> None:
    if not UPSTREAM.is_dir():
        raise SystemExit(f"Pinned CRAFT checkout not found: {UPSTREAM}")
    actual_rev = git_revision(UPSTREAM)
    if actual_rev != UPSTREAM_REV:
        raise SystemExit(f"CRAFT revision mismatch: expected {UPSTREAM_REV}, found {actual_rev}")

    if PATCHED.exists():
        shutil.rmtree(PATCHED)
    shutil.copytree(UPSTREAM, PATCHED, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    runner = PATCHED / "run_craft.py"
    source = runner.read_text(encoding="utf-8")
    original = 'director_order = random.choices(["D1", "D2", "D3"], k=3)'
    replacement = 'director_order = random.sample(["D1", "D2", "D3"], k=3)'
    if source.count(original) != 1:
        raise SystemExit("Expected exactly one upstream Director scheduling line; refusing an unreviewed patch")
    runner.write_text(source.replace(original, replacement), encoding="utf-8")

    # A source-level launch shim fixes the Python RNG before game scheduling.
    # PYTHONHASHSEED is set in the child environment for deterministic archetypes.
    source = runner.read_text(encoding="utf-8")
    anchor = "    RUN           = args.run\n"
    seeded = anchor + "    random.seed(RUN)\n"
    if source.count(anchor) != 1:
        raise SystemExit("Expected exactly one RUN assignment; refusing an unreviewed seed patch")
    runner.write_text(source.replace(anchor, seeded), encoding="utf-8")

    patched_hash = subprocess.run(
        [sys.executable, "-c", "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())", str(runner)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (PATCHED / "tacit_adapter_manifest.json").write_text(
        json.dumps(
            {
                "upstream_revision": UPSTREAM_REV,
                "patch": "seeded sample without replacement for one call per distinct Director; seed global Python RNG from --run",
                "patched_run_craft_py_sha256": patched_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Prepare source and validate endpoint without scored calls")
    args = parser.parse_args()

    check_server()
    prepare_patched_source()
    print(f"Pinned CRAFT source and adapter verified; model endpoint {MODEL} is available.")
    if args.prepare_only:
        return

    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
    env["OPENAI_API_KEY"] = "local-experiment"
    env["PYTHONHASHSEED"] = "317"
    env["PYTHONPATH"] = str(ROOT / ".cache" / "python-packages") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "run_craft.py",
        "--mode", "api",
        "--director", MODEL,
        "--builder", MODEL,
        "--dataset", "data/structures_dataset_20.json",
        "--structures", "0,10,19",
        "--turns", "5",
        "--run", "317",
        "--oracle",
        "--oracle_n", "5",
        "--no_tools",
        "--output", str(RAW / "upstream_results"),
    ]
    log_path = RAW / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=PATCHED,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"Runner exit code: {proc.returncode}; full output: {log_path}")
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
