"""Run a role-separated Private Match v0.2 pilot against local endpoints.

This module never launches a model. Its CLI is dry-run by default and requires
--execute before making requests to a loopback OpenAI-compatible endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tacit.channel import LocalTCPMessageChannel
from tacit.runtime import ChatCompletion, ChatModel, OpenAICompatibleClient

from experiments.emergent_ood_v0_3.runner import (
    _endpoint_port,
    validate_resource_preflight,
)
from experiments.private_match_v0_1.generate_tasks import generate_episode, score_answer
from experiments.private_match_v0_2.protocols import PROTOCOL_IDS, Protocol, protocol_by_id


EXPERIMENT_ID = "private-match-v0.2"
SCORER_ID = "private-match-exact-candidate-id-v1"
MAX_MODEL_CALLS_PER_BATCH = 12
REQUEST_TIMEOUT_SECONDS = 30.0
CAPABILITY_CONDITION = "full_information"


def planned_model_calls(episodes: int, protocols: list[str]) -> int:
    """Count model requests before execution so staged runs have a hard cap."""
    return episodes * sum(1 if name in {"no_message", CAPABILITY_CONDITION} else 2 for name in protocols)


def episode_id_for_seed(seed: int) -> str:
    """Stable episode identity shared across separately executed protocol batches."""
    return f"pm2-{seed:012d}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _receiver_messages(protocol: Protocol, receiver_view: dict[str, Any], message: str | None) -> list[dict[str, str]]:
    feature_order = receiver_view["feature_names"]
    user: dict[str, Any] = {
        "task": "Identify which candidate row is exactly equal to the private record described by the message. Return only its candidate_id.",
        "feature_order": feature_order,
        "candidates": receiver_view["candidates"],
        "message_verbatim": message,
    }
    return [
        {"role": "system", "content": protocol.receiver_instruction},
        {"role": "user", "content": _json(user)},
    ]


def _full_information_messages(
    receiver_view: dict[str, Any], target_record: dict[str, str]
) -> list[dict[str, str]]:
    user = {
        "task": "Identify the candidate whose entire record exactly matches the supplied target record. Return only its candidate_id.",
        "feature_order": receiver_view["feature_names"],
        "target_record": target_record,
        "candidates": receiver_view["candidates"],
    }
    return [
        {
            "role": "system",
            "content": "You are given the exact target record and candidate table. Compare every feature and value, then return only the matching candidate_id.",
        },
        {"role": "user", "content": _json(user)},
    ]


def _call_record(agent: str, stage: str, completion: ChatCompletion, tokenizer_id: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "stage": stage,
        "model": completion.model,
        "tokenizer": tokenizer_id,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "service_seconds": completion.service_seconds,
        "retry": False,
        "truncated": completion.finish_reason == "length",
    }


def _message_diagnostics(
    protocol_id: str, message: str | None, target_record: dict[str, str],
    feature_order: list[str], vocabulary_size: int,
) -> tuple[bool | None, bool | None]:
    """Return (format-valid, semantically-faithful) for mechanically checkable codes."""
    if message is None or protocol_id in {"concise_nl", "autoform"}:
        return None, None
    try:
        if protocol_id == "json":
            decoded = json.loads(message)
            valid = (
                isinstance(decoded, dict)
                and set(decoded) == set(target_record)
                and all(isinstance(value, str) for value in decoded.values())
            )
            return valid, valid and decoded == target_record
        if protocol_id == "tuple":
            values = message.split(",")
            valid = len(values) == len(feature_order) and all(re.fullmatch(r"v\d{4}", value) for value in values)
            decoded = dict(zip(feature_order, values)) if valid else None
            return valid, valid and decoded == target_record
        if protocol_id == "compact_kv":
            parts = message.split(";")
            pairs = [part.split("=", 1) for part in parts if part.count("=") == 1]
            valid = (
                len(parts) == len(feature_order)
                and len(pairs) == len(parts)
                and [pair[0] for pair in pairs] == feature_order
                and all(re.fullmatch(r"v\d{4}", pair[1]) for pair in pairs)
            )
            decoded = {key: value for key, value in pairs} if valid else None
            return valid, valid and decoded == target_record
        if protocol_id == "hex_nibbles":
            valid = len(message) == len(feature_order) and all(ch in "0123456789abcdef" for ch in message)
            numbers = [int(ch, 16) for ch in message] if valid else []
            valid = valid and all(number < vocabulary_size for number in numbers)
            decoded = {name: f"v{number:04d}" for name, number in zip(feature_order, numbers)} if valid else None
            return valid, valid and decoded == target_record
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, False
    return None, None


def run_condition(
    *, episode_id: str, seed: int, candidate_count: int, feature_count: int,
    vocabulary_size: int, protocol: Protocol, sender_model: ChatModel | None,
    receiver_model: ChatModel, sender_tokenizer_id: str | None,
    receiver_tokenizer_id: str, model_population_id: str,
    condition_name: str | None = None,
) -> dict[str, Any]:
    sender_view, receiver_view, gold = generate_episode(
        episode_id=episode_id, seed=seed, candidate_count=candidate_count,
        feature_count=feature_count, vocabulary_size=vocabulary_size,
    )
    calls: list[dict[str, Any]] = []
    receiver_results: list[ChatCompletion] = []
    wall_started = time.perf_counter()
    condition = condition_name or ("no_message" if sender_model is None else protocol.protocol_id)
    if condition == CAPABILITY_CONDITION:
        receiver_results.append(receiver_model.complete(
            _full_information_messages(receiver_view, sender_view["target_record"])
        ))
        answer_output = receiver_results[-1].text
        answer = answer_output.strip()
        transmissions: list[dict[str, Any]] = []
        calls.append(_call_record("receiver", CAPABILITY_CONDITION, receiver_results[-1], receiver_tokenizer_id))
        policy_id = "full_information_capability_control"
        code_id = "direct-target-record-v1"
        decoder_id = "candidate-id-v1"
        message: str | None = None
    elif sender_model is None:
        receiver_results.append(receiver_model.complete(_receiver_messages(protocol, receiver_view, None)))
        answer_output = receiver_results[-1].text
        answer = answer_output.strip()
        transmissions: list[dict[str, Any]] = []
        calls.append(_call_record("receiver", "answer", receiver_results[-1], receiver_tokenizer_id))
        policy_id = "no_message"
        code_id = "none"
        decoder_id = "none"
        message: str | None = None
    else:
        sender_messages = [
            {"role": "system", "content": protocol.sender_instruction},
            {"role": "user", "content": _json({
                "task": "Communicate this private record so another agent can identify its exact row in a candidate table.",
                "feature_order": sender_view["feature_names"],
                "target_record": sender_view["target_record"],
            })},
        ]
        sender_result = sender_model.complete(sender_messages)
        calls.append(_call_record("sender", "encode", sender_result, sender_tokenizer_id or "not_applicable"))
        message = sender_result.text

        def receive(envelope: dict[str, Any]) -> None:
            receiver_results.append(receiver_model.complete(
                _receiver_messages(protocol, receiver_view, envelope["payload"])
            ))

        with LocalTCPMessageChannel(receive) as channel:
            transmission = channel.send(
                message, protocol_id=protocol.protocol_id, round_number=1,
                sender="sender", recipient="receiver",
            )
        if len(receiver_results) != 1:
            raise RuntimeError("message delivery did not invoke the receiver exactly once")
        answer_output = receiver_results[0].text
        answer = answer_output.strip()
        calls.append(_call_record("receiver", "decode", receiver_results[0], receiver_tokenizer_id))
        transmissions = [transmission.cost_record()]
        policy_id, code_id, decoder_id = protocol.protocol_id, protocol.code_id, protocol.decoder_id

    success = score_answer(receiver_view, gold, answer)
    answer_format_valid = (
        answer_output == answer
        and answer in {row["candidate_id"] for row in receiver_view["candidates"]}
    )
    format_valid, message_fidelity = _message_diagnostics(
        condition, message, sender_view["target_record"],
        sender_view["feature_names"], vocabulary_size,
    )
    return {
        "schema_version": "tlu.costs.v3",
        "episode_id": episode_id,
        "stratum": {
            "experiment_id": EXPERIMENT_ID,
            "task_id": "private-record-match-v1",
            "split": "pilot",
            "task_parameters": {
                "candidate_count": candidate_count,
                "feature_count": feature_count,
                "vocabulary_size": vocabulary_size,
            },
            "model_population_id": model_population_id,
            "agent_models": {
                "sender": getattr(sender_model, "model_name", "unknown") if sender_model is not None else "absent",
                "receiver": getattr(receiver_model, "model_name", "unknown"),
            },
            "scorer_id": SCORER_ID,
        },
        "protocol": {"policy_id": policy_id, "code_id": code_id, "decoder_id": decoder_id},
        "outcome": {"joint_success": success, "answer_score": 1.0 if success else 0.0},
        "diagnostics": {
            "condition": condition,
            "generation_seed": seed,
            "message_text": message,
            "answer_text": answer,
            "answer_text_verbatim": answer_output,
            "answer_candidate_id": answer if answer_format_valid else None,
            "answer_format_valid": answer_format_valid,
            "candidate_ids_in_receiver_order": [row["candidate_id"] for row in receiver_view["candidates"]],
            "message_format_valid": format_valid,
            "message_semantic_fidelity": message_fidelity,
            "sender_truncated": bool(calls[0]["truncated"]) if sender_model is not None else None,
            "receiver_truncated": bool(calls[-1]["truncated"]),
        },
        "transmissions": transmissions,
        "model_calls": calls,
        "runtime": {"wall_seconds": time.perf_counter() - wall_started},
        "setup": [],
    }


class _NamedClient:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client
        self.model_name = client.model

    def complete(self, messages: Any) -> ChatCompletion:
        return self.client.complete(messages)


def validate_capability_ledger(
    path: Path | None,
    *,
    episodes: int,
    seed: int,
    candidate_count: int,
    feature_count: int,
    vocabulary_size: int,
    receiver_model: str,
    receiver_tokenizer_id: str,
    model_population_id: str,
) -> None:
    """Require a successful, disjoint calibration block for this task/model pair."""
    if episodes < 1:
        raise ValueError("episode count must be positive")
    if path is None:
        raise ValueError("later protocol batches require --capability-ledger from a 100% calibration full_information batch")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read capability ledger: {path}") from exc
    by_id = {row.get("episode_id"): row for row in rows if isinstance(row, dict)}
    if len(rows) != episodes or len(by_id) != episodes:
        raise ValueError("capability ledger must contain exactly the requested number of calibration episodes")
    evaluation_ids = {episode_id_for_seed(value) for value in range(seed, seed + episodes)}
    calibration_seeds: list[int] = []
    for row in rows:
        diagnostics = row.get("diagnostics") if isinstance(row, dict) else None
        episode_seed = diagnostics.get("generation_seed") if isinstance(diagnostics, dict) else None
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
            raise ValueError("capability ledger generation seeds must be non-negative integers")
        if row.get("episode_id") != episode_id_for_seed(episode_seed):
            raise ValueError("capability ledger episode IDs must match their generation seeds")
        calibration_seeds.append(episode_seed)
    if len(set(calibration_seeds)) != episodes:
        raise ValueError("capability ledger calibration seeds must be unique")
    calibration_seeds.sort()
    if calibration_seeds != list(range(calibration_seeds[0], calibration_seeds[0] + episodes)):
        raise ValueError("capability ledger must contain one contiguous calibration seed block")
    if set(by_id) & evaluation_ids:
        raise ValueError("capability calibration episodes must be disjoint from evaluation episodes to prevent selection bias")
    for episode_seed in calibration_seeds:
        episode_id = episode_id_for_seed(episode_seed)
        row = by_id[episode_id]
        _sender_view, receiver_view, gold = generate_episode(
            episode_id=episode_id,
            seed=episode_seed,
            candidate_count=candidate_count,
            feature_count=feature_count,
            vocabulary_size=vocabulary_size,
        )
        diagnostics = row.get("diagnostics", {})
        calls = row.get("model_calls", [])
        stratum = row.get("stratum", {})
        receiver_calls = [
            call for call in calls
            if isinstance(call, dict) and call.get("agent") == "receiver"
        ]
        if row.get("protocol", {}).get("policy_id") != "full_information_capability_control":
            raise ValueError("capability ledger contains a non-full-information row")
        if stratum.get("experiment_id") != EXPERIMENT_ID or stratum.get("scorer_id") != SCORER_ID:
            raise ValueError("capability ledger experiment or scorer differs from the current run")
        if row.get("transmissions") != [] or len(calls) != 1:
            raise ValueError("full-information capability rows must contain one receiver call and no transmission")
        if row.get("outcome", {}).get("joint_success") is not True:
            raise ValueError("full-information receiver capability screen did not pass every episode")
        if diagnostics.get("condition") != CAPABILITY_CONDITION or diagnostics.get("generation_seed") != episode_seed:
            raise ValueError("capability ledger episode parameters do not match the current run")
        if diagnostics.get("answer_format_valid") is not True:
            raise ValueError("full-information answer must have strict valid candidate-ID format")
        if diagnostics.get("answer_text_verbatim") != gold["target_candidate_id"]:
            raise ValueError("full-information answer does not match the generated target")
        expected_candidates = [row["candidate_id"] for row in receiver_view["candidates"]]
        if diagnostics.get("candidate_ids_in_receiver_order") != expected_candidates:
            raise ValueError("capability ledger candidate table does not match the current task")
        if stratum.get("task_parameters") != {
            "candidate_count": candidate_count,
            "feature_count": feature_count,
            "vocabulary_size": vocabulary_size,
        }:
            raise ValueError("capability ledger task parameters do not match the current run")
        if stratum.get("model_population_id") != model_population_id:
            raise ValueError("capability receiver model population differs from the current run")
        if stratum.get("agent_models", {}).get("receiver") != receiver_model:
            raise ValueError("capability receiver model differs from the current run")
        if (
            len(receiver_calls) != 1
            or receiver_calls[0].get("tokenizer") != receiver_tokenizer_id
            or receiver_calls[0].get("model") != receiver_model
        ):
            raise ValueError("capability receiver model or tokenizer differs from the current run")
        if receiver_calls[0].get("stage") != CAPABILITY_CONDITION or receiver_calls[0].get("truncated") is not False:
            raise ValueError("full-information capability response was missing, wrong-stage, or truncated")


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("only a loopback endpoint URL is allowed; cloud endpoints are disabled")
    return value.rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually call the configured loopback endpoints")
    parser.add_argument("--protocols", nargs="+", choices=(CAPABILITY_CONDITION, "no_message", *PROTOCOL_IDS), default=[CAPABILITY_CONDITION, "no_message", *PROTOCOL_IDS])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--vocabulary-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path(".cache/private_match_v0_2/pilot.jsonl"))
    parser.add_argument("--resource-preflight", type=Path, help="passing same-host report from emergent_ood_v0_3/resource_preflight.ps1, at most five minutes old")
    parser.add_argument("--capability-ledger", type=Path, help="100% full_information ledger required before any protocol comparison")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    parser.add_argument("--sender-model", default=os.environ.get("TLU_SENDER_MODEL", ""))
    parser.add_argument("--receiver-model", default=os.environ.get("TLU_RECEIVER_MODEL", ""))
    parser.add_argument("--sender-tokenizer-id", default=os.environ.get("TLU_SENDER_TOKENIZER_ID", ""))
    parser.add_argument("--receiver-tokenizer-id", default=os.environ.get("TLU_RECEIVER_TOKENIZER_ID", ""))
    parser.add_argument("--model-population-id", default=os.environ.get("TLU_MODEL_POPULATION_ID", "local-unspecified"))
    parser.add_argument("--base-url", default=os.environ.get("TLU_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--sender-base-url", default=os.environ.get("TLU_SENDER_BASE_URL", ""))
    parser.add_argument("--receiver-base-url", default=os.environ.get("TLU_RECEIVER_BASE_URL", ""))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.candidates < 2 or args.features < 1 or args.vocabulary_size < 2:
        parser.error("--candidates must be >=2, --features >=1, and --vocabulary-size >=2")
    if args.vocabulary_size ** args.features < args.candidates:
        parser.error("the requested task has fewer unique records than candidates")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    calls_planned = planned_model_calls(args.episodes, args.protocols)
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run", "would_run": args.episodes * len(args.protocols),
            "planned_model_calls": calls_planned,
            "maximum_model_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
            "fits_call_cap": calls_planned <= MAX_MODEL_CALLS_PER_BATCH,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "protocols": args.protocols, "output": str(args.output),
            "capability_gate_required": any(name != CAPABILITY_CONDITION for name in args.protocols),
            "inference_started": False,
        }, indent=2))
        return 0
    if len(set(args.protocols)) != len(args.protocols):
        parser.error("each condition may appear only once in an execution batch")
    if CAPABILITY_CONDITION in args.protocols and len(args.protocols) != 1:
        parser.error("full_information must run as its own first-stage batch")
    if calls_planned > MAX_MODEL_CALLS_PER_BATCH:
        parser.error(
            f"this batch plans {calls_planned} model calls, above the hard limit of "
            f"{MAX_MODEL_CALLS_PER_BATCH}; split protocols or reduce --episodes"
        )
    needs_sender = any(name not in {"no_message", CAPABILITY_CONDITION} for name in args.protocols)
    if not args.receiver_tokenizer_id or not args.receiver_model or needs_sender and (not args.sender_model or not args.sender_tokenizer_id):
        parser.error("--execute requires receiver model/tokenizer IDs and sender model/tokenizer IDs for message protocols")
    if "hex_nibbles" in args.protocols and (args.features > 16 or args.vocabulary_size > 16):
        parser.error("the registered hex_nibbles condition requires at most 16 features and values")
    if "hex_nibbles" in args.protocols and args.vocabulary_size != 16:
        parser.error("the registered hex_nibbles condition requires vocabulary-size 16")
    sender_endpoint = _loopback_url(args.sender_base_url or args.base_url) if needs_sender else None
    receiver_endpoint = _loopback_url(args.receiver_base_url or args.base_url)
    required_ports = {_endpoint_port(receiver_endpoint)}
    if sender_endpoint is not None:
        required_ports.add(_endpoint_port(sender_endpoint))
    if any(name != CAPABILITY_CONDITION for name in args.protocols):
        try:
            validate_capability_ledger(
                args.capability_ledger,
                episodes=args.episodes,
                seed=args.seed,
                candidate_count=args.candidates,
                feature_count=args.features,
                vocabulary_size=args.vocabulary_size,
                receiver_model=args.receiver_model,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                model_population_id=args.model_population_id,
            )
        except ValueError as exc:
            parser.error(str(exc))
    try:
        validate_resource_preflight(args.resource_preflight, required_ports=required_ports)
    except ValueError as exc:
        parser.error(str(exc))
    target = args.output.resolve()
    if target.exists() and not args.force:
        parser.error(f"refusing to overwrite {target}; pass --force explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    sender = _NamedClient(OpenAICompatibleClient(sender_endpoint, args.sender_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_tokens=64, follow_redirects=False)) if needs_sender and args.sender_model else None
    receiver = _NamedClient(OpenAICompatibleClient(receiver_endpoint, args.receiver_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_tokens=16, follow_redirects=False))
    with target.open("w", encoding="utf-8", newline="\n") as out:
        for index in range(args.episodes):
            for name in args.protocols:
                row = run_condition(
                    episode_id=episode_id_for_seed(args.seed + index), seed=args.seed + index,
                    candidate_count=args.candidates, feature_count=args.features,
                    vocabulary_size=args.vocabulary_size,
                    protocol=protocol_by_id("no_message" if name == CAPABILITY_CONDITION else name),
                    sender_model=None if name in {"no_message", CAPABILITY_CONDITION} else sender,
                    receiver_model=receiver,
                    sender_tokenizer_id=args.sender_tokenizer_id or None,
                    receiver_tokenizer_id=args.receiver_tokenizer_id,
                    model_population_id=args.model_population_id,
                    condition_name=name,
                )
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
    print(json.dumps({
        "mode": "executed", "episodes": args.episodes,
        "model_calls": calls_planned, "output": str(target),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
