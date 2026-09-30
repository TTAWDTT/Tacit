"""Replay frozen usage-transfer messages on compatible counterfactual episodes.

Planning is offline and is the default. Execution reuses frozen sender outputs,
calls only the receiver, and requires the same fresh local-resource and
independent receiver-capability gates as the primary v0.4 runner.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacit.channel import LocalTCPMessageChannel
from tacit.runtime import ChatCompletion, ChatModel, OpenAICompatibleClient
from experiments.emergent_ood_v0_4.runner import (
    CAPABILITY_CALIBRATION_SETS,
    EXPERIMENT_ID,
    MAX_MODEL_CALLS_PER_BATCH,
    SCORER_ID,
    _inside_project,
    _jsonl,
    _loopback_url,
    _endpoint_port,
    load_episode_bundle,
    load_protocol_card,
    load_usage_examples,
    select_candidate_sets,
    validate_capability_ledger,
    validate_resource_preflight,
)
from experiments.emergent_ood_v0_4.split import split_task_id


REPLAY_SCHEMA = "tlu.emergent-ood-message-association-replay.v0.1"
RUN_MANIFEST_SCHEMA = "tlu.emergent-ood-replay-manifest.v0.1"
CHECKPOINT_SCHEMA = "tlu.emergent-ood-replay-checkpoint.v1"
MATCHING_SAMPLER = "uniform-exact-count-dp-v1"
TASK = "Select the candidate ID whose full tuple is the sender's private meaning."
FINAL_INSTRUCTION = "Return only the exact candidate_id of your selected candidate for external scoring."


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _digest(payload)


def _checkpoint_rows_digest(rows: Sequence[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    return _digest(payload)


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path = _inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        **checkpoint,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": _checkpoint_rows_digest(checkpoint["rows"]),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_checkpoint(path: Path, *, run_signature: str) -> dict[str, Any]:
    try:
        checkpoint = json.loads(_inside_project(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("replay checkpoint is missing or invalid") from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("run_signature") != run_signature
        or not isinstance(checkpoint.get("rows"), list)
        or any(not isinstance(row, dict) for row in checkpoint["rows"])
        or checkpoint.get("rows_sha256") != _checkpoint_rows_digest(checkpoint["rows"])
        or not isinstance(checkpoint.get("resource_preflight_sha256s"), list)
        or any(not isinstance(item, str) or len(item) != 64 for item in checkpoint["resource_preflight_sha256s"])
    ):
        raise ValueError("replay checkpoint failed configuration or content-integrity validation")
    if checkpoint["rows"]:
        from tools.cost_report import aggregate

        try:
            aggregate(copy.deepcopy(checkpoint["rows"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("replay checkpoint contains invalid cost rows") from exc
    return checkpoint


def _canonical_tuple(value: Mapping[str, Any], attributes: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, Mapping) or set(value) != set(attributes):
        raise ValueError("source target or candidate meaning does not match the task ontology")
    result = tuple(value[axis] for axis in attributes)
    if any(not isinstance(part, str) for part in result):
        raise ValueError("task meanings must contain only string values")
    return result


def _uniform_perfect_matching(
    edges: Mapping[str, Sequence[str]],
    recipients: Sequence[str],
    rng: random.Random,
) -> tuple[dict[str, str], int]:
    """Sample uniformly from all perfect matchings of a small bipartite graph.

    Exact completion counts let each matching receive probability 1/N. The
    O(n 2^n) dynamic program is deliberately bounded by the runner's 12-call
    batch ceiling; this is not intended as a general large-graph matcher.
    """
    n = len(recipients)
    if n < 1 or n > MAX_MODEL_CALLS_PER_BATCH:
        raise ValueError(
            f"exact uniform matching supports 1..{MAX_MODEL_CALLS_PER_BATCH} episodes per batch"
        )
    if len(set(recipients)) != n or set(edges) != set(recipients):
        raise ValueError("matching graph must contain one edge list per unique receiver")
    donor_ids = sorted({donor for options in edges.values() for donor in options})
    if len(donor_ids) != n:
        raise ValueError("no complete compatible message derangement exists for this batch")
    donor_index = {donor: i for i, donor in enumerate(donor_ids)}
    adjacency = {
        recipient: tuple(sorted({donor_index[d] for d in edges[recipient]}))
        for recipient in recipients
    }

    @lru_cache(maxsize=None)
    def count_from(index: int, used_mask: int) -> int:
        if index == n:
            return 1
        return sum(
            count_from(index + 1, used_mask | (1 << donor))
            for donor in adjacency[recipients[index]]
            if not used_mask & (1 << donor)
        )

    total = count_from(0, 0)
    if total == 0:
        raise ValueError("no complete compatible message derangement exists for this batch")

    assignment: dict[str, str] = {}
    used_mask = 0
    for index, recipient in enumerate(recipients):
        ticket = rng.randrange(count_from(index, used_mask))
        for donor in adjacency[recipient]:
            bit = 1 << donor
            if used_mask & bit:
                continue
            completions = count_from(index + 1, used_mask | bit)
            if ticket < completions:
                assignment[recipient] = donor_ids[donor]
                used_mask |= bit
                break
            ticket -= completions
        else:  # pragma: no cover - guarded by the exact completion count
            raise RuntimeError("uniform matching sampler exhausted a valid completion count")
    return assignment, total


def build_compatible_derangement(
    episodes: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, Any]],
    *,
    attributes: Sequence[str],
    seed: int,
    source_results_sha256: str,
) -> dict[str, str]:
    """Map every receiver episode to one distinct, compatible donor message.

    A donor is compatible only if it is a different episode and its private
    meaning is absent from the recipient's entire candidate table. The
    seeded sampler is uniform over all compatible perfect matchings, and the
    matching preserves the delivered-message multiset exactly.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(source_results_sha256, str) or len(source_results_sha256) != 64:
        raise ValueError("source results SHA-256 is required")
    try:
        bytes.fromhex(source_results_sha256)
    except ValueError as exc:
        raise ValueError("source results SHA-256 must be hexadecimal") from exc
    if len(episodes) < 2 or len(episodes) != len(source_rows):
        raise ValueError("a derangement requires at least two aligned episode/source rows")

    episode_by_id: dict[str, dict[str, Any]] = {}
    row_by_id: dict[str, dict[str, Any]] = {}
    meaning_by_id: dict[str, tuple[str, ...]] = {}
    for episode, row in zip(episodes, source_rows):
        gold = episode.get("gold", {})
        episode_id = gold.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in episode_by_id:
            raise ValueError("episodes must have unique evaluator IDs")
        if row.get("episode_id") != episode_id or row.get("condition") != "usage_only_transfer":
            raise ValueError("source rows must align exactly with usage_only_transfer episodes")
        trace = row.get("trace")
        if not isinstance(trace, dict) or row.get("costs", {}).get("message_delivered") is not True:
            raise ValueError("every source episode must contain a delivered frozen message")
        message = trace.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("source messages must be non-empty strings")
        target = _canonical_tuple(trace.get("target_tuple_for_evaluator"), attributes)
        sender_target = _canonical_tuple(episode.get("sender", {}).get("private_meaning"), attributes)
        if target != sender_target:
            raise ValueError("source trace target does not match its sender episode")
        receiver = episode.get("receiver", {})
        candidates = receiver.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("episode has no receiver candidates")
        candidate_ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
        if len(candidate_ids) != len(candidates) or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("receiver candidate IDs must be complete and unique")
        expected_ids = trace.get("candidate_ids_in_receiver_order")
        if expected_ids != candidate_ids:
            raise ValueError("source receiver candidate order differs from the bound episode")
        candidate_meanings = {
            _canonical_tuple(item.get("attributes"), attributes) for item in candidates
        }
        if target not in candidate_meanings:
            raise ValueError("evaluator target is missing from its receiver candidate table")
        gold_candidate = next(
            (item for item in candidates if item["candidate_id"] == gold.get("candidate_id")), None,
        )
        if gold_candidate is None or _canonical_tuple(gold_candidate.get("attributes"), attributes) != target:
            raise ValueError("evaluator target candidate does not match the sender's private meaning")
        episode_by_id[episode_id] = episode
        row_by_id[episode_id] = row
        meaning_by_id[episode_id] = target

    ids = list(episode_by_id)
    rng = random.Random(f"tlu.message-association-derangement-v1:{seed}:{source_results_sha256}")
    donor_order = list(ids)
    rng.shuffle(donor_order)
    episode_order = list(ids)
    rng.shuffle(episode_order)
    candidate_sets: dict[str, set[tuple[str, ...]]] = {}
    edges: dict[str, list[str]] = {}
    for recipient_id in ids:
        candidate_sets[recipient_id] = {
            _canonical_tuple(candidate["attributes"], attributes)
            for candidate in episode_by_id[recipient_id]["receiver"]["candidates"]
        }
        edges[recipient_id] = [
            donor_id for donor_id in donor_order
            if donor_id != recipient_id and meaning_by_id[donor_id] not in candidate_sets[recipient_id]
        ]
        if not edges[recipient_id]:
            raise ValueError(f"no compatible foreign message exists for receiver episode {recipient_id}")

    recipient_to_donor, _matching_count = _uniform_perfect_matching(
        edges, episode_order, rng,
    )
    if set(recipient_to_donor) != set(ids) or len(set(recipient_to_donor.values())) != len(ids):
        raise RuntimeError("internal matching error: result is not a bijection")
    for recipient_id, donor_id in recipient_to_donor.items():
        if donor_id == recipient_id or meaning_by_id[donor_id] in candidate_sets[recipient_id]:
            raise RuntimeError("internal matching error: incompatible edge in assignment")
    return recipient_to_donor


def receiver_request(
    *,
    message: str,
    candidates: Sequence[dict[str, Any]],
    protocol_card: dict[str, str],
    attributes: Sequence[str],
    values: Mapping[str, Sequence[str]],
    usage_examples: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the original runner's receiver prompt with only its message changed."""
    from experiments.emergent_ood_v0_4.runner import _Protocol

    protocol = _Protocol(
        "usage_only_transfer", attributes, values,
        protocol_card=protocol_card, usage_examples=usage_examples,
    )
    context = {
        "candidates": list(candidates),
        "training_examples": [
            {"meaning": example["meaning"], "message": example["message"]}
            for example in usage_examples
        ],
    }
    return [
        {
            "role": "system",
            "content": protocol.agent_instructions["receiver"] + "\n\n" + FINAL_INSTRUCTION,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": TASK,
                    "private_context": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    "visible_transcript": [{"sender": "sender", "message": message}],
                    "instruction": "Return the final task answer for external scoring.",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _parse_choice(text: str, candidate_ids: Sequence[str]) -> tuple[str | None, bool]:
    answer = text.strip()
    return (answer, True) if answer in candidate_ids else (None, False)


def run_replay(
    *,
    episodes: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, Any]],
    receiver_model: ChatModel,
    attributes: Sequence[str],
    values: Mapping[str, Sequence[str]],
    protocol_card: dict[str, str],
    usage_examples: Sequence[dict[str, Any]],
    receiver_tokenizer_id: str,
    model_population_id: str,
    stage: str,
    split_seed: int,
    task_seed: int,
    task_id: str,
    ontology_id: str | None,
    target_support_size: int,
    wire_budget_bytes: int,
    seed: int,
    source_results_sha256: str,
    completed_rows: Sequence[dict[str, Any]] = (),
    on_row_complete: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Call only the receiver on a hash-bound compatible message permutation."""
    assignment = build_compatible_derangement(
        episodes, source_rows, attributes=attributes, seed=seed,
        source_results_sha256=source_results_sha256,
    )
    episode_by_id = {episode["gold"]["episode_id"]: episode for episode in episodes}
    row_by_id = {row["episode_id"]: row for row in source_rows}
    results_by_id: dict[str, dict[str, Any]] = {
        row["episode_id"]: copy.deepcopy(row) for row in completed_rows
    }
    expected_prefix = [episode["gold"]["episode_id"] for episode in episodes[:len(results_by_id)]]
    def is_valid_completed_row(row: dict[str, Any]) -> bool:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in episode_by_id:
            return False
        donor_id = assignment[episode_id]
        donor_message = row_by_id[donor_id]["trace"]["message"]
        candidates = episode_by_id[episode_id]["receiver"]["candidates"]
        expected_target = next(
            item["attributes"] for item in candidates
            if item["candidate_id"] == episode_by_id[episode_id]["gold"]["candidate_id"]
        )
        trace = row.get("trace", {})
        return (
            row.get("condition") == "usage_only_transfer_message_deranged"
            and trace.get("source_episode_id") == donor_id
            and trace.get("message") == donor_message
            and trace.get("source_message_sha256") == _digest(donor_message.encode("utf-8"))
            and trace.get("candidate_ids_in_receiver_order") == [item["candidate_id"] for item in candidates]
            and trace.get("target_tuple_for_evaluator") == expected_target
            and row.get("costs", {}).get("model_call_count") == 1
        )

    if list(results_by_id) != expected_prefix or any(
        not is_valid_completed_row(row) for row in results_by_id.values()
    ):
        raise ValueError("completed replay rows are not an exact prefix of this frozen assignment")
    channel = LocalTCPMessageChannel(lambda _: None)
    for episode in episodes:
        recipient_id = episode["gold"]["episode_id"]
        donor_id = assignment[recipient_id]
        if recipient_id in results_by_id:
            continue
        episode = episode_by_id[recipient_id]
        source_row = row_by_id[donor_id]
        message = source_row["trace"]["message"]
        source_message_protocol_id = source_row["protocol_id"]
        transmission = channel.measure(
            message, protocol_id=source_message_protocol_id, round_number=1,
            sender="sender", recipient="receiver",
        )
        if transmission.total_application_bytes > wire_budget_bytes:
            raise ValueError(f"counterfactual message exceeds receiver communication budget: {recipient_id}")
        prompt = receiver_request(
            message=message,
            candidates=episode["receiver"]["candidates"],
            protocol_card=protocol_card,
            attributes=attributes,
            values=values,
            usage_examples=usage_examples,
        )
        call_started = time.perf_counter()
        completion = receiver_model.complete(prompt)
        wall_seconds = time.perf_counter() - call_started
        candidate_ids = [item["candidate_id"] for item in episode["receiver"]["candidates"]]
        answer_id, answer_valid = _parse_choice(completion.text, candidate_ids)
        gold = episode["gold"]
        target_row = next(
            item for item in episode["receiver"]["candidates"]
            if item["candidate_id"] == gold["candidate_id"]
        )
        call_record = {
            "agent": "receiver",
            "round": 1,
            "stage": "counterfactual_replay",
            "model": completion.model,
            "tokenizer": receiver_tokenizer_id,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "service_seconds": completion.service_seconds,
            "retry": False,
            "truncated": completion.finish_reason == "length",
        }
        results_by_id[recipient_id] = {
            "schema_version": "tlu.costs.v3",
            "schema": REPLAY_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "inference_cluster_id": f"split={split_seed}",
            "candidate_set_cluster_id": gold["candidate_set_id"],
            "stage": stage,
            "split_seed": split_seed,
            "ontology_id": ontology_id,
            "task_seed": task_seed,
            "episode_id": recipient_id,
            "candidate_set_id": gold["candidate_set_id"],
            "meaning_id": gold["meaning_id"],
            "condition": "usage_only_transfer_message_deranged",
            "communication_budget_bytes": wire_budget_bytes,
            "protocol_id": source_row["protocol_id"],
            "outcome": {
                "answer_format_valid": answer_valid,
                "answer_candidate_id": answer_id,
                "target_candidate_id": gold["candidate_id"],
                "exact_selection": answer_id == gold["candidate_id"],
                "joint_success": answer_id == gold["candidate_id"],
                "answer_score": float(answer_id == gold["candidate_id"]),
                "bayes_no_message_reference": 1 / len(candidate_ids),
            },
            "protocol": {
                "policy_id": "usage_only_transfer_message_deranged",
                "code_id": source_row["protocol_id"],
                "decoder_id": SCORER_ID,
            },
            "costs": {
                "model_calls": [call_record],
                "model_call_count": 1,
                "complete_input_tokens": completion.input_tokens,
                "complete_output_tokens": completion.output_tokens,
                "calls_with_input_token_usage": int(completion.input_tokens is not None),
                "calls_with_output_token_usage": int(completion.output_tokens is not None),
                "generated_message_bytes": len(message.encode("utf-8")),
                "message_delivered": True,
                "delivered_payload_bytes": transmission.logical_payload_bytes,
                "application_wire_bytes": transmission.total_application_bytes,
                "complete_reported_service_seconds": completion.service_seconds,
                "calls_with_service_time": int(completion.service_seconds is not None),
                "wall_seconds": wall_seconds,
                "communication_budget_bytes": wire_budget_bytes,
                "counterfactual_message_replay": True,
                "sender_generation_reused_from_source_run": True,
                "usage_example_count": len(usage_examples),
                "usage_example_bytes_per_receiver_request": len(json.dumps(
                    [{"meaning": example["meaning"], "message": example["message"]}
                     for example in usage_examples],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")),
            },
            "trace": {
                "message": message,
                "answer": completion.text,
                "candidate_ids_in_receiver_order": candidate_ids,
                "target_tuple_for_evaluator": target_row["attributes"],
                "source_episode_id": donor_id,
                "source_message_sha256": _digest(message.encode("utf-8")),
                "receiver_episode_id": recipient_id,
                "counterfactual": True,
            },
            "stratum": {
                "experiment_id": EXPERIMENT_ID,
                "task_id": task_id,
                "split": stage,
                "scorer_id": SCORER_ID,
                "model_population_id": model_population_id,
                "agent_models": {
                    "sender": source_row.get("stratum", {}).get("agent_models", {}).get(
                        "sender", "frozen_source_sender_unreported",
                    ),
                    "receiver": completion.model,
                },
                "task_parameters": {
                    "candidate_count": len(candidate_ids),
                    "target_support_size": target_support_size,
                    "communication_budget_bytes": wire_budget_bytes,
                },
            },
            # This is the measured cost of the substituted counterfactual
            # channel message, not an additional sender call in this replay.
            "transmissions": [transmission.cost_record(recipient_tokenizer=receiver_tokenizer_id)],
            "model_calls": [call_record],
            "runtime": {"wall_seconds": wall_seconds},
            "setup": copy.deepcopy(source_row.get("setup", [])),
        }
        if on_row_complete is not None:
            on_row_complete(copy.deepcopy(results_by_id[recipient_id]))
    # Restore the episode order independently of matching traversal order.
    rows = [results_by_id[episode["gold"]["episode_id"]] for episode in episodes]
    return rows, assignment


def _write_output(rows: Sequence[dict[str, Any]], path: Path, manifest: dict[str, Any]) -> str:
    path = _inside_project(path)
    manifest_path = _inside_project(path.with_suffix(path.suffix + ".manifest.json"))
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    digest = _digest(data)
    complete = {
        "schema": RUN_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
        "results_file": path.name,
        "results_sha256": digest,
        "result_records": len(rows),
        "contains_evaluator_only_labels": True,
        "interpretation": "receiver-only counterfactual diagnostic; sender generation cost is reused from source and must not be treated as a standalone deployment cost",
    }
    temporary_data = path.with_suffix(path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_data.write_bytes(data)
    temporary_manifest.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_data.replace(path)
    temporary_manifest.replace(manifest_path)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--stage", choices=("validation", "test"), default="test")
    parser.add_argument("--sets", type=int, required=True)
    parser.add_argument("--set-offset", type=int, default=0)
    parser.add_argument("--source-results", type=Path, nargs="+", required=True,
                        help="one or more completed usage_only_transfer JSONL runs covering this batch")
    parser.add_argument("--protocol-card", type=Path, required=True)
    parser.add_argument("--usage-examples", type=Path, required=True)
    parser.add_argument("--usage-reuse-horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume a matching atomic receiver-replay checkpoint")
    parser.add_argument("--max-replay-calls", type=int, help="explicit maximum receiver requests; execution requires an exact batch ceiling")
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--capability-ledger", type=Path)
    parser.add_argument("--capability-input-dir", type=Path)
    parser.add_argument("--capability-split-seed", type=int)
    parser.add_argument("--base-url", default=os.environ.get("TLU_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--receiver-base-url", default=os.environ.get("TLU_RECEIVER_BASE_URL", ""))
    parser.add_argument("--receiver-model", default=os.environ.get("TLU_RECEIVER_MODEL", ""))
    parser.add_argument("--receiver-tokenizer-id", default=os.environ.get("TLU_RECEIVER_TOKENIZER_ID", ""))
    parser.add_argument("--model-population-id", default=os.environ.get("TLU_MODEL_POPULATION_ID", "local-unspecified"))
    args = parser.parse_args()
    if args.sets < 1 or args.set_offset < 0 or args.usage_reuse_horizon < 1:
        parser.error("sets and usage reuse horizon must be positive; set offset must be non-negative")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if args.resume and not args.execute:
        parser.error("--resume is valid only with --execute")

    try:
        bundle, split = load_episode_bundle(args.input_dir, split_seed=args.split_seed)
        episodes = select_candidate_sets(bundle, args.stage, args.sets, set_offset=args.set_offset)
        card, card_digest = load_protocol_card(args.protocol_card)
        input_manifest_digest = _digest((_inside_project(args.input_dir) / "manifest.json").read_bytes())
        examples, examples_digest, _ = load_usage_examples(
            args.usage_examples, split=split, bundle=bundle, protocol_card=card,
            protocol_card_sha256=card_digest,
            training_episode_manifest_sha256=input_manifest_digest,
        )
        expected_ids = [episode["gold"]["episode_id"] for episode in episodes]
        expected_id_set = set(expected_ids)
        source_row_by_id: dict[str, dict[str, Any]] = {}
        source_artifacts: list[dict[str, str]] = []
        source_manifests: list[dict[str, Any]] = []
        source_settings = (
            "communication_budget_bytes", "sender_model", "receiver_model",
            "sender_tokenizer_id", "receiver_tokenizer_id", "model_population_id",
            "temperature", "sender_max_tokens", "receiver_max_tokens", "receiver_endpoint",
        )
        source_settings_value: dict[str, Any] | None = None
        for source_argument in args.source_results:
            source_path = _inside_project(source_argument)
            source_bytes = source_path.read_bytes()
            source_file_digest = _digest(source_bytes)
            rows_in_file = _jsonl(source_path)
            source_manifest_path = _inside_project(source_path.with_suffix(source_path.suffix + ".manifest.json"))
            manifest_bytes = source_manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != "tlu.emergent-ood-run-manifest.v0.4"
                or manifest.get("results_file") != source_path.name
                or manifest.get("results_sha256") != source_file_digest
                or manifest.get("result_records") != len(rows_in_file)
            ):
                raise ValueError(f"source usage-transfer results or manifest hash is invalid: {source_path.name}")
            base_fields = {
                "experiment_id": EXPERIMENT_ID,
                "stage": args.stage,
                "conditions": ["usage_only_transfer"],
                "split_seed": args.split_seed,
                "input_episode_manifest_sha256": input_manifest_digest,
                "protocol_card_sha256": card_digest,
                "usage_examples_sha256": examples_digest,
                "usage_reuse_horizon": args.usage_reuse_horizon,
                "split_sha256": split["split_sha256"],
                "ontology_id": split.get("ontology_id"),
                "task_seed": bundle["manifest"]["task_seed"],
                "candidate_count": bundle["manifest"]["k"],
                "task_id": split_task_id(split),
            }
            for key, expected in base_fields.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"source run manifest does not match replay input ({key}): {source_path.name}")
            first_offset = manifest.get("candidate_set_offset")
            source_set_count = manifest.get("candidate_sets")
            if (
                type(first_offset) is not int or type(source_set_count) is not int
                or source_set_count < 1
                or first_offset < args.set_offset
                or first_offset + source_set_count > args.set_offset + args.sets
                or manifest.get("communication_budget_bytes") is None
            ):
                raise ValueError(f"source run candidate-set range does not lie within this replay batch: {source_path.name}")
            if type(manifest.get("communication_budget_bytes")) is not int or manifest["communication_budget_bytes"] < 0:
                raise ValueError("source run communication budget is malformed")
            settings = {key: manifest.get(key) for key in source_settings}
            if source_settings_value is None:
                source_settings_value = settings
            elif settings != source_settings_value:
                raise ValueError("source runs differ in model, tokenizer, population, generation settings, or byte budget")
            source_episodes = select_candidate_sets(
                bundle, args.stage, source_set_count, set_offset=first_offset,
            )
            source_ids = [episode["gold"]["episode_id"] for episode in source_episodes]
            source_episode_by_id = {
                episode["gold"]["episode_id"]: episode for episode in source_episodes
            }
            indexed_rows = {row.get("episode_id"): row for row in rows_in_file}
            if (
                len(indexed_rows) != len(rows_in_file) or len(rows_in_file) != len(source_episodes)
                or set(indexed_rows) != set(source_ids) or not set(source_ids).issubset(expected_id_set)
            ):
                raise ValueError(f"source results do not cover their declared complete candidate sets: {source_path.name}")
            if set(indexed_rows) & set(source_row_by_id):
                raise ValueError("source results overlap episode IDs; each frozen sender output must occur once")
            for episode_id, row in indexed_rows.items():
                calls = row.get("model_calls")
                gold = source_episode_by_id[episode_id]["gold"]
                costs = row.get("costs")
                outcome = row.get("outcome")
                stratum = row.get("stratum")
                trace = row.get("trace", {})
                source_message = trace.get("message") if isinstance(trace, dict) else None
                transmissions = row.get("transmissions")
                transmission = transmissions[0] if isinstance(transmissions, list) and len(transmissions) == 1 else None
                payload_metadata = transmission.get("payload_metadata") if isinstance(transmission, dict) else None
                if (
                    not isinstance(costs, dict) or not isinstance(outcome, dict) or not isinstance(stratum, dict)
                    or row.get("experiment_id") != EXPERIMENT_ID or row.get("stage") != args.stage
                    or row.get("split_seed") != args.split_seed
                    or row.get("task_seed") != bundle["manifest"]["task_seed"]
                    or row.get("candidate_set_id") != gold["candidate_set_id"]
                    or row.get("meaning_id") != gold["meaning_id"]
                    or outcome.get("target_candidate_id") != gold["candidate_id"]
                    or costs.get("message_delivered") is not True
                    or row.get("protocol_id") != card["protocol_id"]
                    or stratum.get("model_population_id") != manifest.get("model_population_id")
                    or stratum.get("task_id") != split_task_id(split)
                    or costs.get("communication_budget_bytes") != manifest.get("communication_budget_bytes")
                    or not isinstance(calls, list) or len(calls) != 2
                    or any(not isinstance(call, dict) for call in calls)
                    or [call.get("agent") for call in calls] != ["sender", "receiver"]
                    or [call.get("stage") for call in calls] != ["dialogue_turn", "final_answer"]
                    or any(call.get("truncated") is not False for call in calls)
                    or calls[0].get("model") != manifest.get("sender_model")
                    or calls[0].get("tokenizer") != manifest.get("sender_tokenizer_id")
                    or calls[1].get("model") != manifest.get("receiver_model")
                    or calls[1].get("tokenizer") != manifest.get("receiver_tokenizer_id")
                    or not isinstance(source_message, str) or not source_message.strip()
                    or costs.get("generated_message_bytes") != len(source_message.encode("utf-8"))
                    or costs.get("delivered_payload_bytes") != len(source_message.encode("utf-8"))
                    or transmission is None
                    or not isinstance(payload_metadata, dict)
                    or transmission.get("sender") != "sender"
                    or transmission.get("recipients") != ["receiver"]
                    or transmission.get("round") != 1
                    or payload_metadata.get("protocol_id") != card["protocol_id"]
                    or payload_metadata.get("logical_text_utf8_bytes") != len(source_message.encode("utf-8"))
                    or costs.get("application_wire_bytes") != transmission.get("payload_bytes", -1) + transmission.get("framing_bytes", 0)
                    or row.get("trace", {}).get("transmissions") != transmissions
                ):
                    raise ValueError(f"source row is not a delivered complete message from its declared run: {episode_id}")
                source_row_by_id[episode_id] = row
            source_artifacts.append({
                "results_file": source_path.name,
                "results_sha256": source_file_digest,
                "manifest_sha256": _digest(manifest_bytes),
            })
            source_manifests.append(manifest)
        if set(source_row_by_id) != expected_id_set:
            raise ValueError("source result artifacts must cover exactly the selected replay episode batch")
        source_rows = [source_row_by_id[episode_id] for episode_id in expected_ids]
        source_digest = _digest(json.dumps(
            sorted(item["results_sha256"] for item in source_artifacts), separators=(",", ":"),
        ).encode("utf-8"))
        source_manifest = source_manifests[0]
        assignment = build_compatible_derangement(
            episodes, source_rows, attributes=split["attributes"], seed=args.seed,
            source_results_sha256=source_digest,
        )
        receiver_endpoint = _loopback_url(args.receiver_base_url or args.base_url)
        output = _inside_project(args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    planned_calls = len(episodes)
    if planned_calls > MAX_MODEL_CALLS_PER_BATCH:
        parser.error(f"replay batch plans {planned_calls} receiver calls; hard cap is {MAX_MODEL_CALLS_PER_BATCH}")
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "schema": REPLAY_SCHEMA,
            "stage": args.stage,
            "episodes": len(episodes),
            "receiver_calls_planned": planned_calls,
            "maximum_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
            "compatible_derangement_complete": True,
            "messages_reused_once_each": True,
            "matching_sampler": MATCHING_SAMPLER,
            "source_results_sha256": source_digest,
            "source_result_artifacts": source_artifacts,
            "protocol_card_sha256": card_digest,
            "usage_examples_sha256": examples_digest,
            "seed": args.seed,
            "recipient_to_donor_episode_ids": assignment,
            "output": str(output),
            "model_loaded": False,
            "inference_started": False,
        }, indent=2, sort_keys=True))
        return 0

    if args.max_replay_calls != planned_calls:
        parser.error("execution requires --max-replay-calls equal to the exact planned replay count")
    if not args.receiver_model or not args.receiver_tokenizer_id:
        parser.error("--execute requires --receiver-model and --receiver-tokenizer-id")
    if (
        args.receiver_model != source_manifest.get("receiver_model")
        or args.receiver_tokenizer_id != source_manifest.get("receiver_tokenizer_id")
        or args.model_population_id != source_manifest.get("model_population_id")
        or receiver_endpoint != source_manifest.get("receiver_endpoint")
        or source_manifest.get("temperature") != 0.0
        or source_manifest.get("receiver_max_tokens") != 48
    ):
        parser.error("receiver model, tokenizer, population, temperature, or output limit differs from the frozen source run")
    if args.capability_input_dir is None or args.capability_split_seed is None:
        parser.error("execution requires an independent receiver capability bundle and split seed")
    try:
        calibration_bundle, calibration_split = load_episode_bundle(
            args.capability_input_dir, split_seed=args.capability_split_seed,
        )
        if args.capability_split_seed == args.split_seed:
            raise ValueError("capability split seed must differ from replay split seed")
        for key in ("task_key_id", "task_seed", "k"):
            if calibration_bundle["manifest"][key] != bundle["manifest"][key]:
                raise ValueError("capability bundle does not match replay task key, seed, or candidate count")
        if calibration_split["attributes"] != split["attributes"] or calibration_split["values_by_attribute"] != split["values_by_attribute"]:
            raise ValueError("capability bundle ontology differs from replay ontology")
        calibration_episodes = select_candidate_sets(
            calibration_bundle, "train", CAPABILITY_CALIBRATION_SETS,
        )
        validate_capability_ledger(
            args.capability_ledger,
            expected_calibration_episodes=calibration_episodes,
            evaluation_episodes=episodes,
            input_manifest_sha256=_digest((_inside_project(args.capability_input_dir) / "manifest.json").read_bytes()),
            split_seed=args.capability_split_seed,
            split_sha256=calibration_split["split_sha256"],
            evaluation_split_seed=args.split_seed,
            task_seed=bundle["manifest"]["task_seed"],
            task_key_id=bundle["manifest"]["task_key_id"],
            receiver_model=args.receiver_model,
            receiver_tokenizer_id=args.receiver_tokenizer_id,
            model_population_id=args.model_population_id,
            task_id=split_task_id(split),
            train_target_support_size=len(split["train_meaning_ids"]),
            ontology_id=split.get("ontology_id"),
        )
        preflight_path = _inside_project(args.resource_preflight) if args.resource_preflight else None
        validate_resource_preflight(
            preflight_path, required_ports={_endpoint_port(receiver_endpoint)},
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    checkpoint_path = _inside_project(output.with_suffix(output.suffix + ".checkpoint.json"))
    output_manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    current_preflight_sha256 = _digest(preflight_path.read_bytes())
    run_config = {
        "schema": REPLAY_SCHEMA,
        "stage": args.stage,
        "episode_ids": expected_ids,
        "input_episode_manifest_sha256": input_manifest_digest,
        "source_result_artifacts": source_artifacts,
        "source_results_sha256": source_digest,
        "protocol_card_sha256": card_digest,
        "usage_examples_sha256": examples_digest,
        "usage_reuse_horizon": args.usage_reuse_horizon,
        "recipient_to_donor_episode_ids": assignment,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "task_seed": bundle["manifest"]["task_seed"],
        "receiver_endpoint": receiver_endpoint,
        "receiver_model": args.receiver_model,
        "receiver_tokenizer_id": args.receiver_tokenizer_id,
        "model_population_id": args.model_population_id,
        "temperature": 0.0,
        "receiver_max_tokens": 48,
        "wire_budget_bytes": source_manifest["communication_budget_bytes"],
        "max_replay_calls": args.max_replay_calls,
        "matching_sampler": MATCHING_SAMPLER,
    }
    run_signature = _signature(run_config)
    try:
        if output.exists() or output_manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite replay output: {output}")
        if args.resume:
            checkpoint = _read_checkpoint(checkpoint_path, run_signature=run_signature)
            if checkpoint.get("run_config") != run_config:
                raise ValueError("replay checkpoint frozen configuration does not match this batch")
            checkpoint_rows = checkpoint["rows"]
            if len(checkpoint_rows) > len(expected_ids):
                raise ValueError("replay checkpoint has more rows than the planned batch")
            for index, row in enumerate(checkpoint_rows):
                episode_id = expected_ids[index]
                if (
                    row.get("episode_id") != episode_id
                    or row.get("condition") != "usage_only_transfer_message_deranged"
                    or row.get("trace", {}).get("source_episode_id") != assignment[episode_id]
                ):
                    raise ValueError("replay checkpoint rows are not an exact prefix of the frozen assignment")
            if current_preflight_sha256 not in checkpoint["resource_preflight_sha256s"]:
                checkpoint["resource_preflight_sha256s"].append(current_preflight_sha256)
            resumed = True
            reused_rows = len(checkpoint_rows)
            _write_checkpoint(checkpoint_path, checkpoint)
        else:
            if checkpoint_path.exists():
                raise FileExistsError(f"replay checkpoint already exists; pass --resume to continue: {checkpoint_path}")
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "run_signature": run_signature,
                "run_config": run_config,
                "rows": [],
                "resource_preflight_sha256s": [current_preflight_sha256],
                "resumed": False,
            }
            checkpoint_rows = checkpoint["rows"]
            resumed = False
            reused_rows = 0
            _write_checkpoint(checkpoint_path, checkpoint)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    client = OpenAICompatibleClient(
        receiver_endpoint, args.receiver_model, timeout_seconds=45.0,
        max_tokens=48, follow_redirects=False, temperature=0.0,
    )
    try:
        def checkpoint_row(row: dict[str, Any]) -> None:
            checkpoint["rows"].append(row)
            _write_checkpoint(checkpoint_path, checkpoint)

        rows, final_assignment = run_replay(
            episodes=episodes, source_rows=source_rows, receiver_model=client,
            attributes=split["attributes"], values=split["values_by_attribute"],
            protocol_card=card, usage_examples=examples,
            receiver_tokenizer_id=args.receiver_tokenizer_id,
            model_population_id=args.model_population_id, stage=args.stage,
            split_seed=args.split_seed, task_seed=bundle["manifest"]["task_seed"],
            task_id=split_task_id(split), ontology_id=split.get("ontology_id"),
            target_support_size=bundle["manifest"]["partition_sizes"][args.stage],
            wire_budget_bytes=source_manifest.get("communication_budget_bytes", 4096),
            seed=args.seed, source_results_sha256=source_digest,
            completed_rows=checkpoint_rows,
            on_row_complete=checkpoint_row,
        )
        if final_assignment != assignment:
            raise RuntimeError("deterministic replay assignment changed between validation and execution")
        replay_manifest = {
            "experiment_id": EXPERIMENT_ID,
            "stage": args.stage,
            "conditions": ["usage_only_transfer_message_deranged"],
            "candidate_sets": args.sets,
            "candidate_set_offset": args.set_offset,
            "split_seed": args.split_seed,
            "split_sha256": split["split_sha256"],
            "task_seed": bundle["manifest"]["task_seed"],
            "input_episode_manifest_sha256": input_manifest_digest,
            "source_results_sha256": source_digest,
            "source_manifest_sha256s": [item["manifest_sha256"] for item in source_artifacts],
            "protocol_card_sha256": card_digest,
            "usage_examples_sha256": examples_digest,
            "receiver_model": args.receiver_model,
            "receiver_tokenizer_id": args.receiver_tokenizer_id,
            "model_population_id": args.model_population_id,
            "resource_preflight_sha256": _digest(preflight_path.read_bytes()),
            "resource_preflight_sha256s": checkpoint["resource_preflight_sha256s"],
            "max_replay_calls": args.max_replay_calls,
            "run_signature": run_signature,
            "resumed_from_checkpoint": resumed,
            "checkpoint_rows_reused": reused_rows,
            "recipient_to_donor_episode_ids": assignment,
            "matching_sampler": MATCHING_SAMPLER,
            "seed": args.seed,
            "receiver_only_cost_scope": True,
        }
        output_digest = _write_output(rows, output, replay_manifest)
        checkpoint_path.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"receiver replay failed; no completed result manifest is available: {exc}")
    print(json.dumps({
        "mode": "executed",
        "results_sha256": output_digest,
        "result_records": len(rows),
        "receiver_calls": sum(row["costs"]["model_call_count"] for row in rows),
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
