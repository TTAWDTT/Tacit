"""Select a natural-language template using only the frozen development split.

The selector consumes completed, resource-gated development ledgers. It does
not call models, inspect evaluation outcomes, or alter prompts. Every candidate
call is charged as optimizer setup in the emitted freeze manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.private_match_v0_3.generate_tasks import _validate_task_key, task_key_id
from experiments.private_match_v0_3.protocols import PROMPT_REVISION
from experiments.private_match_v0_3.report import validate_private_match_records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read development ledger {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"development ledger rows must be JSON objects: {path}")
    return rows


def _call_signature(records: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    signature = None
    for row in records:
        current = tuple(
            (call.get("agent"), call.get("model"), call.get("tokenizer"))
            for call in row["model_calls"]
        )
        if any(not agent or not model or not tokenizer for agent, model, tokenizer in current):
            raise ValueError("natural-language selection requires model and tokenizer IDs on every call")
        if signature is None:
            signature = current
        elif current != signature:
            raise ValueError("model or tokenizer signature changed within a candidate ledger")
    return signature or ()


def _candidate_summary(protocol_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    payload_bytes = 0
    faithful_messages = 0
    message_count = 0
    call_count = 0
    wall_seconds = 0.0
    successes = 0
    for row in records:
        successes += row["outcome"]["joint_success"] is True
        wall_seconds += float(row.get("runtime", {}).get("wall_seconds", 0.0))
        for call in row["model_calls"]:
            for field in ("input_tokens", "output_tokens"):
                value = call.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"selection requires complete non-negative {field} telemetry")
            input_tokens += call["input_tokens"]
            output_tokens += call["output_tokens"]
            call_count += 1
        for transmission in row["transmissions"]:
            payload_bytes += transmission["payload_metadata"]["logical_text_utf8_bytes"]
        for message in row["diagnostics"]["message_diagnostics"]:
            message_count += 1
            faithful_messages += message["semantic_fidelity"] is True
    return {
        "protocol_id": protocol_id,
        "development_episodes": len(records),
        "joint_successes": successes,
        "joint_success_fraction": f"{successes}/{len(records)}",
        "faithful_messages": faithful_messages,
        "message_count": message_count,
        "model_calls": call_count,
        "model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "complete_model_tokens": input_tokens + output_tokens,
        "logical_payload_bytes": payload_bytes,
        "wall_seconds": wall_seconds,
    }


def select_natural_language_baseline(
    *, ledger_paths: dict[str, Path], task_key: bytes,
) -> dict[str, Any]:
    """Select among preregistered NL candidates using disjoint dev records only."""
    _validate_task_key(task_key)
    prereg_path = Path(__file__).with_name("preregistration.json")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    pilot = prereg["default_pilot"]
    freeze = prereg["protocol_freeze"]
    candidates = tuple(freeze["development_natural_language_candidates"])
    if set(ledger_paths) != set(candidates) or len(candidates) < 2:
        raise ValueError("one development ledger is required for every frozen NL candidate")
    if freeze.get("revision") != PROMPT_REVISION:
        raise ValueError("protocol code revision differs from preregistration")

    development_seeds = tuple(
        range(pilot["development_seed_start"],
              pilot["development_seed_start"] + pilot["development_episodes"])
    )
    calibration_seeds = set(range(pilot["calibration_seed_start"],
                                  pilot["calibration_seed_start"] + pilot["calibration_episodes"]))
    evaluation_seeds = set(range(pilot["evaluation_seed_start"],
                                 pilot["evaluation_seed_start"] + pilot["evaluation_episodes"]))
    if calibration_seeds.intersection(development_seeds) or evaluation_seeds.intersection(development_seeds):
        raise ValueError("development seeds must be disjoint from calibration and evaluation")
    expected_ids = {f"pmt3-{seed:012d}" for seed in development_seeds}

    summaries: dict[str, dict[str, Any]] = {}
    signatures = set()
    digests: dict[str, str] = {}
    task_key_ids = set()
    task_shapes = set()
    model_populations = set()
    for protocol_id in candidates:
        path = ledger_paths[protocol_id].resolve()
        records = _load_jsonl(path)
        validate_private_match_records(records, task_key=task_key)
        if len(records) != len(development_seeds):
            raise ValueError(f"{protocol_id} ledger must contain every development episode exactly once")
        if {row.get("episode_id") for row in records} != expected_ids:
            raise ValueError(f"{protocol_id} ledger is not the preregistered development shard")
        for row in records:
            stratum = row["stratum"]
            if stratum.get("split") != "development" or row["diagnostics"].get("split") != "development":
                raise ValueError("natural-language selection accepts development records only")
            if row["diagnostics"].get("condition") != "both_sources":
                raise ValueError("all development candidates must use the both_sources condition")
            if row["protocol"].get("representation_id") != protocol_id:
                raise ValueError("ledger protocol ID does not match the candidate file")
            task_key_ids.add(stratum["task_parameters"]["task_key_id"])
            params = stratum["task_parameters"]
            task_shapes.add((params["q"], params["candidate_count"], params["agent_count"]))
            model_populations.add(stratum.get("model_population_id"))
        signatures.add(_call_signature(records))
        summaries[protocol_id] = _candidate_summary(protocol_id, records)
        digests[protocol_id] = _sha256(path)
    if len(signatures) != 1:
        raise ValueError("candidate ledgers must use identical model and tokenizer IDs")
    if task_key_ids != {task_key_id(task_key)}:
        raise ValueError("development ledgers use inconsistent evaluator task keys")
    if task_shapes != {(pilot["q"], pilot["candidate_count"], pilot["agent_count"])}:
        raise ValueError("development candidate ledgers must use the preregistered task shape")
    if len(model_populations) != 1 or None in model_populations:
        raise ValueError("development candidates must use the same model population")
    episode_sets = [
        {row["episode_id"] for row in _load_jsonl(Path(ledger_paths[name]).resolve())}
        for name in candidates
    ]
    if any(episode_set != episode_sets[0] for episode_set in episode_sets[1:]):
        raise ValueError("candidate protocols must be paired on identical episodes")

    ordered = sorted(
        summaries.values(),
        key=lambda item: (
            -item["joint_successes"],
            item["complete_model_tokens"],
            item["logical_payload_bytes"],
            item["protocol_id"],
        ),
    )
    setup = {
        "candidate_protocols": len(candidates),
        "development_model_calls": sum(item["model_calls"] for item in summaries.values()),
        "development_model_input_tokens": sum(item["model_input_tokens"] for item in summaries.values()),
        "development_model_output_tokens": sum(item["model_output_tokens"] for item in summaries.values()),
        "development_logical_payload_bytes": sum(item["logical_payload_bytes"] for item in summaries.values()),
        "development_wall_seconds": sum(item["wall_seconds"] for item in summaries.values()),
        "account_as_optimizer_setup": True,
    }
    protocols_path = Path(__file__).with_name("protocols.py")
    return {
        "schema_version": "tlu.private-match-nl-selection.v1",
        "selection_split": "development_only",
        "selected_protocol_id": ordered[0]["protocol_id"],
        "prompt_revision": PROMPT_REVISION,
        "selection_rule": freeze["development_selection"],
        "development_episode_ids": sorted(expected_ids),
        "development_seed_start": pilot["development_seed_start"],
        "development_episodes": pilot["development_episodes"],
        "task_key_id": task_key_id(task_key),
        "candidate_metrics": [summaries[name] for name in candidates],
        "candidate_ledger_sha256": digests,
        "protocols_py_sha256": _sha256(protocols_path),
        "preregistration_sha256": _sha256(prereg_path),
        "model_and_tokenizer_signature": [list(item) for item in next(iter(signatures))],
        "optimizer_setup_cost": setup,
        "evaluation_data_used": False,
        "limits": [
            "selection is limited to the two preregistered English templates",
            "development success is used for selection and is not confirmatory evidence",
            "all candidate development calls and payload costs count as setup",
            "this is not an open-ended or LLM-generated prompt search",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-key-file", type=Path, required=True)
    parser.add_argument("--concise-ledger", type=Path, required=True)
    parser.add_argument("--short-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists() and not args.force:
        parser.error(f"refusing to overwrite {target}; pass --force explicitly")
    try:
        task_key = args.task_key_file.read_bytes()
        result = select_natural_language_baseline(
            ledger_paths={
                "concise_nl": args.concise_ledger,
                "short_nl": args.short_ledger,
            },
            task_key=task_key,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selection_manifest": str(target),
                      "selected_protocol_id": result["selected_protocol_id"],
                      "optimizer_setup_cost": result["optimizer_setup_cost"]}, indent=2))


if __name__ == "__main__":
    main()
