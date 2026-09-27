"""Run the frozen three-structure CRAFT natural-language reference baseline."""

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
PATCHED = ROOT / ".cache" / "research" / "CRAFT-tacit-v0_22"
RAW = ROOT / ".cache" / "pilot_v0_22"
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
            models = json.load(response)
        with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
            slots = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Local llama.cpp server unavailable: {exc}") from exc
    ids = {item.get("id") for item in models.get("data", [])}
    if MODEL not in ids:
        raise SystemExit(f"Expected model alias {MODEL!r}; got {sorted(ids)}")
    if len(slots) != 1 or slots[0].get("n_ctx") != 4096:
        raise SystemExit(f"Expected exactly one 4096-token slot; got {[(s.get('n_ctx')) for s in slots]}")


def prepare_source() -> None:
    PATCHED.resolve().relative_to(ROOT.resolve())
    actual_revision = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != UPSTREAM_REV:
        raise SystemExit(f"CRAFT source must be pinned at {UPSTREAM_REV}")
    if PATCHED.exists():
        shutil.rmtree(PATCHED)
    shutil.copytree(UPSTREAM, PATCHED, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    runner = PATCHED / "run_craft.py"
    source = runner.read_text(encoding="utf-8")
    old_schedule = 'director_order = random.choices(["D1", "D2", "D3"], k=3)'
    if source.count(old_schedule) != 1:
        raise SystemExit("Unexpected upstream Director scheduler; refusing to patch")
    source = source.replace(old_schedule, 'director_order = random.sample(["D1", "D2", "D3"], k=3)')
    seed_anchor = "    RUN           = args.run\n"
    if source.count(seed_anchor) != 1:
        raise SystemExit("Unexpected upstream run-seed location; refusing to patch")
    runner.write_text(source.replace(seed_anchor, seed_anchor + "    random.seed(RUN)\n"), encoding="utf-8")

    director = PATCHED / "agents" / "director_agent.py"
    source = director.read_text(encoding="utf-8")
    start = source.index("    def parse_director_response(self, response_text):")
    end = source.index("\n    def parse_director_response_gemini", start)
    method = '''    def parse_director_response(self, response_text):
        """Extract only public message content and remove its outer format wrappers."""
        response_text = response_text or ""
        think_match = re.search(r"<think>\\s*(.*?)\\s*</think>", response_text, re.DOTALL | re.IGNORECASE)
        message_match = re.search(
            r"<message>\\s*(.*?)(?:</message>|$)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if message_match:
            public_message = message_match.group(1).strip()
        else:
            visible = response_text
            if think_match:
                visible = response_text[think_match.end():]
            elif re.search(r"<think>", response_text, re.IGNORECASE):
                visible = ""
            bracketed = re.findall(r"\\[([^\\[\\]]+)\\]", visible, re.DOTALL)
            bracketed = [
                part.strip()
                for part in bracketed
                if part.strip() and "natural human speech only" not in part.lower()
            ]
            public_message = "\\n".join(bracketed) if bracketed else visible.strip()

        public_message = re.sub(r"</?message>", "", public_message, flags=re.IGNORECASE).strip()
        if len(public_message) >= 2 and public_message.startswith("[") and public_message.endswith("]"):
            public_message = public_message[1:-1].strip()
        if not public_message or "natural human speech only" in public_message.lower():
            public_message = "No message provided"
        return {
            "internal_thinking": think_match.group(1).strip() if think_match else "No thinking provided",
            "public_message": public_message,
            "raw_response": response_text,
        }
'''
    director.write_text(source[:start] + method + source[end:], encoding="utf-8")

    (PATCHED / "tacit_adapter_manifest.json").write_text(
        json.dumps(
            {
                "source_revision": UPSTREAM_REV,
                "model_sha256": MODEL_SHA256,
                "server": {"context": 4096, "parallel": 1, "batch_size": 1024, "ubatch_size": 256},
                "patches": [
                    "seeded distinct-role schedule",
                    "message-region extraction, wrapper/tag removal, otherwise preserve public prose",
                    "UTF-8 child I/O",
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
    print("Pinned CRAFT source, model, server context, and message parser verified.")
    if args.prepare_only:
        return

    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
    env["OPENAI_API_KEY"] = "local-experiment"
    env["PYTHONHASHSEED"] = "322"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / ".cache" / "python-packages") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "run_craft.py",
        "--mode", "api",
        "--director", MODEL,
        "--builder", MODEL,
        "--dataset", "data/structures_dataset_20.json",
        "--structures", "0,7,2",
        "--turns", "8",
        "--run", "322",
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
