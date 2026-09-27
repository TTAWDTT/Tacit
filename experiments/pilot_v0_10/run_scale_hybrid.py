"""Run the v0.10 Qwen3-8B paired hybrid role diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V08 = ROOT / "experiments" / "pilot_v0_8"
sys.path.insert(0, str(V08))

import run_prefixsum as runner  # noqa: E402
from run_oracle_control import oracle_call_llm  # noqa: E402

PREREG = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
VARIANTS = {
    "oracle_sender_qwen_receiver": 0,
    "qwen_sender_oracle_receiver": 1,
}
REPORT_JSON = ROOT / "research" / "PREFIXSUM_MODEL_SCALE_V0_10.json"


def _agent_id(messages: list[dict[str, Any]]) -> int:
    import re

    initial = next(
        message["content"] for message in messages
        if message.get("role") == "user" and "Your private segment" in message.get("content", "")
    )
    match = re.search(r"You are Agent (\d+) of 2", initial)
    if not match:
        raise ValueError("Unable to identify agent in model request")
    return int(match.group(1))


def main() -> None:
    if PREREG["task_manifest_sha256"] != runner.POLICIES["task_manifest_sha256"]:
        raise SystemExit("Preregistered task manifest differs from the v0.8 runner")
    actual_upstream = subprocess.check_output(
        ["git", "-C", str(runner.UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_upstream != runner.POLICIES["upstream_engine_commit"]:
        raise SystemExit(f"Pinned upstream mismatch: {actual_upstream}")

    runner.POLICIES["model_file_sha256"] = PREREG["model_file_sha256"]
    runner.POLICIES["model_file_size_bytes"] = PREREG["model_file_size_bytes"]
    runner.POLICIES["model_revision"] = PREREG["model_repository_revision"]
    runner.POLICIES["runtime"] = PREREG["runtime"]
    runner.MODEL_NAME = "Qwen3-8B-Q4_K_M"
    runner.OUTPUT = ROOT / ".cache" / "pilot_v0_10"
    runner.OUTPUT.mkdir(parents=True, exist_ok=True)
    runner._verify_task_manifest()
    runner._verify_model_file()
    runner._warm_local_model()

    original_call_llm = runner.engine.call_llm
    rows: list[dict[str, Any]] = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_index = runner.OUTPUT / f"scale_hybrid_runs_{run_id}.jsonl"
    try:
        for task_index, task_name in enumerate(runner.TASK_FILES):
            order = list(VARIANTS)
            if task_index % 2:
                order.reverse()
            for variant in order:
                oracle_agent_id = VARIANTS[variant]

                def hybrid_call_llm(*, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
                    if _agent_id(messages) == oracle_agent_id:
                        return oracle_call_llm(messages=messages, **kwargs)
                    return original_call_llm(messages=messages, **kwargs)

                runner.engine.call_llm = hybrid_call_llm
                print(f"Running {task_name} / {variant}", flush=True)
                row = runner.run_one(task_name, "compact_kv")
                row["variant"] = variant
                rows.append(row)
                with trace_index.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        runner.engine.call_llm = original_call_llm

    subprocess.run(
        [sys.executable, str(HERE / "analyze_scale_hybrid.py"), "--runs", str(trace_index)],
        check=True,
    )


if __name__ == "__main__":
    main()
