"""Run the frozen three-level bounded-history CRAFT baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache/research/CRAFT"
PATCHED = ROOT / ".cache/research/CRAFT-tacit-v0_23"
RAW = ROOT / ".cache/pilot_v0_23"
MODEL_PATH = ROOT / ".cache/models/Qwen3-8B-Q4_K_M.gguf"
REV = "f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430"
MODEL = "Qwen3-8B-Q4_K_M"
MODEL_BYTES = 5027783488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"


def verify_model() -> None:
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise SystemExit("Qwen3-8B model size/path mismatch")
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"Qwen3-8B checksum mismatch: {digest}")


def check_server() -> None:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        models = json.load(response)
    with urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5) as response:
        slots = json.load(response)
    if MODEL not in {item.get("id") for item in models.get("data", [])}:
        raise SystemExit("Wrong model alias on local server")
    if len(slots) != 1 or slots[0].get("n_ctx") != 4096:
        raise SystemExit("Expected one local slot with context 4096")


def patch_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"Unexpected {label} source; refusing to patch")
    return source.replace(old, new)


def prepare_source() -> None:
    PATCHED.resolve().relative_to(ROOT.resolve())
    revision = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REV:
        raise SystemExit(f"CRAFT revision mismatch: {revision}")
    if PATCHED.exists():
        shutil.rmtree(PATCHED)
    shutil.copytree(UPSTREAM, PATCHED, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    runner = PATCHED / "run_craft.py"
    source = runner.read_text(encoding="utf-8")
    source = patch_once(
        source,
        'director_order = random.choices(["D1", "D2", "D3"], k=3)',
        'director_order = random.sample(["D1", "D2", "D3"], k=3)',
        "Director scheduler",
    )
    source = patch_once(
        source,
        '            public_conversation = "\\n".join(conversation_history)',
        '            public_conversation = "\\n".join(conversation_history[-16:])',
        "shared-history window",
    )
    source = patch_once(source, "    RUN           = args.run\n", "    RUN           = args.run\n    random.seed(RUN)\n", "run seed")
    start = source.index('                if(resp["internal_thinking"].__contains__("429")):\n')
    end = source.index("                director_responses[did] = resp\n", start)
    error_handling = '''                if (
                    not resp.get("raw_response")
                    and str(resp.get("internal_thinking", "")).startswith("Error in generation:")
                ):
                    turn_data.setdefault("director_backend_errors", []).append(
                        {"director": did, "error": str(resp.get("internal_thinking", ""))}
                    )
                    resp["public_message"] = "No message provided"
                    director_responses[did] = resp
                    continue

'''
    source = source[:start] + error_handling + source[end:]
    runner.write_text(source, encoding="utf-8")

    director = PATCHED / "agents/director_agent.py"
    source = director.read_text(encoding="utf-8")
    start = source.index("    def parse_director_response(self, response_text):")
    end = source.index("\n    def parse_director_response_gemini", start)
    parser_method = '''    def parse_director_response(self, response_text):
        """Extract public text and remove the message's format wrappers."""
        response_text = response_text or ""
        think = re.search(r"<think>\\s*(.*?)\\s*</think>", response_text, re.S | re.I)
        message = re.search(r"<message>\\s*(.*?)(?:</message>|$)", response_text, re.S | re.I)
        if message:
            public = message.group(1).strip()
        else:
            visible = response_text[think.end():] if think else response_text
            if not think and re.search(r"<think>", response_text, re.I):
                visible = ""
            bracketed = re.findall(r"\\[([^\\[\\]]+)\\]", visible, re.S)
            bracketed = [x.strip() for x in bracketed if x.strip() and "natural human speech only" not in x.lower()]
            public = "\\n".join(bracketed) if bracketed else visible.strip()
        public = re.sub(r"</?message>", "", public, flags=re.I).strip()
        if len(public) >= 2 and public.startswith("[") and public.endswith("]"):
            public = public[1:-1].strip()
        if not public or "natural human speech only" in public.lower():
            public = "No message provided"
        return {
            "internal_thinking": think.group(1).strip() if think else "No thinking provided",
            "public_message": public,
            "raw_response": response_text,
        }
'''
    director.write_text(source[:start] + parser_method + source[end:], encoding="utf-8")
    (PATCHED / "tacit_adapter_manifest.json").write_text(
        json.dumps(
            {"source_revision": REV, "model_sha256": MODEL_SHA256, "history_lines": 16,
             "server": {"context": 4096, "parallel": 1, "batch_size": 1024, "ubatch_size": 256},
             "patches": ["distinct role schedule", "bounded history", "clean message parser",
                         "failed calls omitted from channel", "remove substring 429 exit", "UTF-8"]},
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
    print("Pinned CRAFT/model/runtime and adapters verified.")
    if args.prepare_only:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OPENAI_BASE_URL": "http://127.0.0.1:8000/v1", "OPENAI_API_KEY": "local-experiment",
                "PYTHONHASHSEED": "323", "PYTHONIOENCODING": "utf-8"})
    env["PYTHONPATH"] = str(ROOT / ".cache/python-packages") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "run_craft.py", "--mode", "api", "--director", MODEL, "--builder", MODEL,
               "--dataset", "data/structures_dataset_20.json", "--structures", "0,7,2", "--turns", "8",
               "--run", "323", "--oracle", "--oracle_n", "5", "--no_tools", "--output",
               str(RAW / "upstream_results")]
    with (RAW / "runner.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PATCHED, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"Runner exit code: {result.returncode}; log: {RAW / 'runner.log'}")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
