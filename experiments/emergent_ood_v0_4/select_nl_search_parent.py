"""Freeze an exact-score training frontier and choose the next NL-search parent.

This utility accepts one hash-bound feedback artifact per candidate, rebuilds
each artifact from its source train ledgers/results, and requires the candidates
to be paired over identical training episodes and model settings. Its output is
search state only; it is not a validation freeze or a test result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.emergent_ood_v0_4.episodes import PROJECT_ROOT
from experiments.emergent_ood_v0_4.nl_feedback import FEEDBACK_SCHEMA, build_feedback, _project_path


SEARCH_SCHEMA = "tlu.emergent-ood-nl-search-frontier.v0.1"


def _read_feedback(path: Path) -> tuple[dict[str, Any], bytes]:
    resolved = _project_path(path)
    payload = resolved.read_bytes()
    if len(payload) > 1_048_576:
        raise ValueError(f"feedback artifact exceeds 1 MiB: {path}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"feedback artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != FEEDBACK_SCHEMA:
        raise ValueError(f"unsupported feedback artifact: {path}")
    expected = build_feedback(
        episode_dir=PROJECT_ROOT / value["input_episode_dir"],
        run_paths=[PROJECT_ROOT / item["results"] for item in value["source_runs"]],
        card_path=PROJECT_ROOT / value["source_card"],
        induction_manifest_path=PROJECT_ROOT / value["source_induction_manifest"],
        split_seed=value["split_seed"],
    )
    canonical = lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if canonical(expected) != canonical(value):
        raise ValueError("feedback does not match its hash-bound source train run")
    return value, payload


def _dominates(left: dict[str, Any], right: dict[str, Any], dimensions: list[str]) -> bool:
    weakly_better = True
    strictly_better = False
    for dimension in dimensions:
        lv, rv = left[dimension], right[dimension]
        if dimension == "exact_success_rate":
            if lv < rv:
                weakly_better = False
            strictly_better |= lv > rv
        else:
            if lv > rv:
                weakly_better = False
            strictly_better |= lv < rv
    return weakly_better and strictly_better


def freeze_training_frontier(feedback_paths: list[Path]) -> dict[str, Any]:
    candidates = []
    payloads = []
    for path in feedback_paths:
        feedback, payload = _read_feedback(path)
        if feedback.get("split") != "train" or feedback.get("protocol_family") != "plain_english":
            raise ValueError("search parent selection only accepts training-stage plain-English feedback")
        if feedback.get("validation_files_opened") is not False or feedback.get("test_files_opened") is not False:
            raise ValueError("feedback artifact does not certify train-only construction")
        candidates.append({
            "protocol_id": feedback["protocol_id"],
            "feedback_path": _project_path(path).relative_to(PROJECT_ROOT).as_posix(),
            "feedback_sha256": hashlib.sha256(payload).hexdigest(),
            "source_induction_manifest_sha256": feedback["source_induction_manifest_sha256"],
            "optimization_round": feedback["optimization_round"],
            "max_optimization_rounds": feedback["max_optimization_rounds"],
            "candidate_count": feedback["candidate_count"],
            "episodes": feedback["episodes"],
            "exact_successes": feedback["exact_successes"],
            "exact_success_rate": feedback["exact_success_rate"],
            "invalid_answers": feedback["invalid_answers"],
            "application_wire_bytes": feedback["application_wire_bytes"],
            "model_calls": feedback["model_calls"],
            "complete_input_tokens": feedback["complete_input_tokens"],
            "complete_output_tokens": feedback["complete_output_tokens"],
            "complete_service_seconds": feedback["complete_service_seconds"],
            "wall_seconds": feedback["wall_seconds"],
            "source_induction_model_calls": feedback.get("source_induction_model_calls"),
            "source_induction_input_tokens": feedback.get("source_induction_input_tokens"),
            "source_induction_output_tokens": feedback.get("source_induction_output_tokens"),
            "source_induction_prompt_completion_utf8_bytes": feedback.get("source_induction_prompt_completion_utf8_bytes"),
        })
        payloads.append(feedback)

    ids = [row["protocol_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate feedback artifacts contain duplicate protocol IDs")
    proposal_manifests = {feedback["source_induction_manifest_sha256"] for feedback in payloads}
    if len(proposal_manifests) != 1:
        raise ValueError("one search round must use candidates from exactly one induction manifest")
    source_manifest_path = _project_path(PROJECT_ROOT / payloads[0]["source_induction_manifest"])
    source_manifest = json.loads(source_manifest_path.read_bytes())
    declared_candidate_count = source_manifest.get("candidate_count")
    source_cards = source_manifest.get("candidate_cards")
    if (
        isinstance(declared_candidate_count, bool)
        or not isinstance(declared_candidate_count, int)
        or not isinstance(source_cards, list)
        or len(source_cards) != declared_candidate_count
        or len(candidates) != declared_candidate_count
    ):
        raise ValueError("search frontier requires feedback for every card in the frozen induction candidate set")
    expected_card_hashes = {
        item.get("protocol_id"): item.get("sha256") for item in source_cards if isinstance(item, dict)
    }
    observed_card_hashes = {
        feedback["protocol_id"]: feedback["protocol_card_sha256"] for feedback in payloads
    }
    if len(expected_card_hashes) != len(source_cards) or observed_card_hashes != expected_card_hashes:
        raise ValueError("feedback card identities/hashes do not cover the complete induction candidate set")
    pairing_keys = (
        "split_seed", "split_sha256", "input_episode_manifest_sha256", "training_episode_ids_sha256",
        "task_key_id", "task_seed", "model_population_id", "sender_model", "receiver_model",
        "sender_tokenizer_id", "receiver_tokenizer_id", "communication_budget_bytes", "episodes",
        "optimization_round", "max_optimization_rounds", "candidate_count",
    )
    for key in pairing_keys:
        if len({json.dumps(feedback.get(key), sort_keys=True) for feedback in payloads}) != 1:
            raise ValueError(f"candidate feedback is not paired: {key} differs")

    dimensions = ["exact_success_rate", "application_wire_bytes"]
    if all(row["complete_input_tokens"] is not None for row in candidates):
        dimensions.append("complete_input_tokens")
    if all(row["complete_output_tokens"] is not None for row in candidates):
        dimensions.append("complete_output_tokens")
    if all(row["complete_service_seconds"] is not None for row in candidates):
        dimensions.append("complete_service_seconds")
    pareto_ids = [
        candidate["protocol_id"] for candidate in candidates
        if not any(_dominates(other, candidate, dimensions) for other in candidates if other is not candidate)
    ]
    # Exact-score quality is the primary search objective. Costs break ties so
    # the next round improves the strongest observed parent without hiding the
    # full nondominated quality/cost frontier in the report.
    def champion_key(item: dict[str, Any]) -> tuple[Any, ...]:
        token_sum = (
            item["complete_input_tokens"] + item["complete_output_tokens"]
            if item["complete_input_tokens"] is not None and item["complete_output_tokens"] is not None
            else math.inf
        )
        service = item["complete_service_seconds"] if item["complete_service_seconds"] is not None else math.inf
        return (-item["exact_success_rate"], item["application_wire_bytes"], token_sum, service, item["protocol_id"])

    champion = min(candidates, key=champion_key)
    # Proposal cost is charged once per unique induced candidate set, not once
    # per candidate card from that same generator call.
    proposal_ledgers: dict[str, dict[str, Any]] = {}
    for feedback in payloads:
        proposal_id = feedback["source_induction_manifest_sha256"]
        proposal_ledgers.setdefault(proposal_id, {
            "model_calls": feedback.get("source_induction_model_calls"),
            "input_tokens": feedback.get("source_induction_input_tokens"),
            "output_tokens": feedback.get("source_induction_output_tokens"),
            "prompt_completion_utf8_bytes": feedback.get("source_induction_prompt_completion_utf8_bytes"),
        })
    setup = {"candidate_card_proposal_ledgers": list(proposal_ledgers.values())}
    for metric in ("model_calls", "input_tokens", "output_tokens", "prompt_completion_utf8_bytes"):
        values = [item[metric] for item in setup["candidate_card_proposal_ledgers"]]
        setup[metric] = sum(values) if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) else None
    setup["candidate_evaluation_model_calls"] = sum(item["model_calls"] for item in candidates)
    setup["candidate_evaluation_application_wire_bytes"] = sum(item["application_wire_bytes"] for item in candidates)
    setup["candidate_evaluation_input_tokens"] = (
        sum(item["complete_input_tokens"] for item in candidates)
        if all(item["complete_input_tokens"] is not None for item in candidates) else None
    )
    setup["candidate_evaluation_output_tokens"] = (
        sum(item["complete_output_tokens"] for item in candidates)
        if all(item["complete_output_tokens"] is not None for item in candidates) else None
    )
    setup["candidate_evaluation_service_seconds"] = (
        sum(item["complete_service_seconds"] for item in candidates)
        if all(item["complete_service_seconds"] is not None for item in candidates) else None
    )
    setup["candidate_evaluation_wall_seconds"] = sum(item["wall_seconds"] for item in candidates)
    exhausted = champion["optimization_round"] >= champion["max_optimization_rounds"]
    return {
        "schema": SEARCH_SCHEMA,
        "selection_stage": "training_search_only",
        "selection_rule": "retain nondominated candidates; next-round parent maximizes exact training success, then minimizes application wire bytes, complete input+output tokens, service seconds, then protocol ID",
        "candidate_metrics": candidates,
        "pareto_dimensions": dimensions,
        "pareto_protocol_ids": pareto_ids,
        "next_round_parent_protocol_id": champion["protocol_id"],
        "next_round_feedback_path": champion["feedback_path"],
        "next_round": None if exhausted else champion["optimization_round"] + 1,
        "search_budget_exhausted": exhausted,
        "paired_split_seed": payloads[0]["split_seed"],
        "paired_split_sha256": payloads[0]["split_sha256"],
        "paired_training_episode_ids_sha256": payloads[0]["training_episode_ids_sha256"],
        "model_population_id": payloads[0]["model_population_id"],
        "search_setup_cost": setup,
        "validation_data_used": False,
        "test_data_used": False,
        "limits": [
            "Training frontier and chosen parent are optimization state, not confirmatory results.",
            "The quality-first parent rule may choose a larger message when exact success improves; the full Pareto set remains available for the later validation freeze.",
            "Candidate proposal cost is counted once per unique induction manifest; evaluation cost sums each paired candidate run.",
            "The frozen iteration budget is not extended automatically; a larger search requires a new preregistration and fresh held-out evaluation design.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, action="append", required=True, help="one feedback artifact per candidate; repeat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _project_path(args.output)
        if output.exists():
            raise ValueError(f"refusing to overwrite existing search frontier: {output}")
        result = freeze_training_frontier(args.feedback)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "search_frontier": output.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "candidates": len(result["candidate_metrics"]),
        "pareto_protocol_ids": result["pareto_protocol_ids"],
        "next_round_parent_protocol_id": result["next_round_parent_protocol_id"],
        "validation_data_used": False,
        "test_data_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
