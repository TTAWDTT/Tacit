"""Run the preregistered CRAFT v0.24 communication-necessity ablation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.craft_v0_23 import run_craft_baseline as base

RAW = ROOT / ".cache/pilot_v0_24"
MAX_TURNS = 20
STRUCTURES = "0,7,2"


def replace_once(source: str, old: str, new: str, what: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"Expected one {what} anchor, found {source.count(old)}")
    return source.replace(old, new)


def make_condition_tree(name: str, no_directors: bool, random_oracle: bool) -> Path:
    source_tree = base.PATCHED
    target = ROOT / f".cache/research/CRAFT-tacit-v0_24-{name}"
    target.resolve().relative_to(ROOT.resolve())
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_tree, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    runner = target / "run_craft.py"
    source = runner.read_text(encoding="utf-8")
    if no_directors:
        start = source.index('            director_order = random.sample(["D1", "D2", "D3"], k=3)\n')
        end_anchor = "            turn_data['director_responses'] = director_responses\n"
        end = source.index(end_anchor, start) + len(end_anchor)
        no_message_block = '''            # v0.24 intervention: no Director calls and no public messages.
            director_order = []
            director_responses = {}
            turn_data['director_responses'] = director_responses
'''
        source = source[:start] + no_message_block + source[end:]
    runner.write_text(source, encoding="utf-8")

    if random_oracle:
        builder = target / "agents/builder_agent.py"
        code = builder.read_text(encoding="utf-8")
        code = replace_once(code, "import random\n", "import random\n_TACTIC_ORACLE_RNG = random.Random(240323)\n", "oracle RNG")
        code = replace_once(
            code,
            "        try:\n            prompt = self.create_builder_prompt(director_discussion, current_state, available_blocks, oracle_moves=oracle_moves,  )\n",
            "        try:\n            if oracle_moves is not None:\n                if oracle_moves:\n                    return dict(_TACTIC_ORACLE_RNG.choice(oracle_moves))\n                return {\"action\": \"clarify\", \"clarification\": \"No oracle candidates available.\"}\n            prompt = self.create_builder_prompt(director_discussion, current_state, available_blocks, oracle_moves=oracle_moves,  )\n",
            "random oracle policy insertion",
        )
        builder.write_text(code, encoding="utf-8")
    return target


def run_condition(tree: Path, name: str) -> int:
    output = RAW / name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        "OPENAI_API_KEY": "local-experiment",
        "PYTHONHASHSEED": "323",
        "PYTHONIOENCODING": "utf-8",
    })
    env["PYTHONPATH"] = str(ROOT / ".cache/python-packages") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable, "run_craft.py", "--mode", "api", "--director", base.MODEL,
        "--builder", base.MODEL, "--dataset", "data/structures_dataset_20.json",
        "--structures", STRUCTURES, "--turns", str(MAX_TURNS), "--run", "323",
        "--oracle", "--oracle_n", "5", "--no_tools", "--output", str(output),
    ]
    log_path = RAW / f"{name}.runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"{name}: exit={result.returncode}; output={output}; log={log_path}", flush=True)
    return result.returncode


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["all", "natural_language_team", "zero_message_llm_builder", "zero_message_random_oracle_policy"], default="all")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    base.verify_model()
    base.check_server()
    base.prepare_source()
    print("Pinned CRAFT/model/runtime and v0.23 adapters verified.", flush=True)
    if args.prepare_only:
        return

    conditions = [
        ("natural_language_team", False, False),
        ("zero_message_llm_builder", True, False),
        ("zero_message_random_oracle_policy", True, True),
    ]
    for name, no_directors, random_oracle in conditions:
        if args.condition not in {"all", name}:
            continue
        tree = make_condition_tree(name, no_directors, random_oracle)
        code = run_condition(tree, name)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
