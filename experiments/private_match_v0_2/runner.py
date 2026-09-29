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

from experiments.private_match_v0_1.generate_tasks import generate_episode, score_answer
from experiments.private_match_v0_2.protocols import PROTOCOL_IDS, Protocol, protocol_by_id


EXPERIMENT_ID = "private-match-v0.2"
SCORER_ID = "private-match-exact-candidate-id-v1"


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
) -> dict[str, Any]:
    sender_view, receiver_view, gold = generate_episode(
        episode_id=episode_id, seed=seed, candidate_count=candidate_count,
        feature_count=feature_count, vocabulary_size=vocabulary_size,
    )
    calls: list[dict[str, Any]] = []
    receiver_results: list[ChatCompletion] = []
    wall_started = time.perf_counter()
    if sender_model is None:
        receiver_results.append(receiver_model.complete(_receiver_messages(protocol, receiver_view, None)))
        answer = receiver_results[-1].text.strip()
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
        answer = receiver_results[0].text.strip()
        calls.append(_call_record("receiver", "decode", receiver_results[0], receiver_tokenizer_id))
        transmissions = [transmission.cost_record()]
        policy_id, code_id, decoder_id = protocol.protocol_id, protocol.code_id, protocol.decoder_id

    success = score_answer(receiver_view, gold, answer)
    format_valid, message_fidelity = _message_diagnostics(
        protocol.protocol_id, message, sender_view["target_record"],
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
            "generation_seed": seed,
            "message_text": message,
            "answer_text": answer,
            "answer_is_candidate_id": answer in {row["candidate_id"] for row in receiver_view["candidates"]},
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


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("only a loopback endpoint URL is allowed; cloud endpoints are disabled")
    return value.rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually call the configured loopback endpoints")
    parser.add_argument("--protocols", nargs="+", choices=(*PROTOCOL_IDS, "no_message"), default=["no_message", *PROTOCOL_IDS])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--vocabulary-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path(".cache/private_match_v0_2/pilot.jsonl"))
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
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run", "would_run": args.episodes * len(args.protocols),
            "protocols": args.protocols, "output": str(args.output),
            "inference_started": False,
        }, indent=2))
        return 0
    if not args.receiver_tokenizer_id or not args.receiver_model or any(p != "no_message" for p in args.protocols) and (not args.sender_model or not args.sender_tokenizer_id):
        parser.error("--execute requires receiver model/tokenizer IDs and sender model/tokenizer IDs for message protocols")
    if "hex_nibbles" in args.protocols and (args.features > 16 or args.vocabulary_size > 16):
        parser.error("the registered hex_nibbles condition requires at most 16 features and values")
    if "hex_nibbles" in args.protocols and args.vocabulary_size != 16:
        parser.error("the registered hex_nibbles condition requires vocabulary-size 16")
    sender_endpoint = _loopback_url(args.sender_base_url or args.base_url)
    receiver_endpoint = _loopback_url(args.receiver_base_url or args.base_url)
    target = args.output.resolve()
    if target.exists() and not args.force:
        parser.error(f"refusing to overwrite {target}; pass --force explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    sender = _NamedClient(OpenAICompatibleClient(sender_endpoint, args.sender_model, max_tokens=64, follow_redirects=False)) if args.sender_model else None
    receiver = _NamedClient(OpenAICompatibleClient(receiver_endpoint, args.receiver_model, max_tokens=16, follow_redirects=False))
    with target.open("w", encoding="utf-8", newline="\n") as out:
        for index in range(args.episodes):
            for name in args.protocols:
                row = run_condition(
                    episode_id=f"pm2-{index:06d}", seed=args.seed + index,
                    candidate_count=args.candidates, feature_count=args.features,
                    vocabulary_size=args.vocabulary_size,
                    protocol=protocol_by_id(name),
                    sender_model=None if name == "no_message" else sender,
                    receiver_model=receiver,
                    sender_tokenizer_id=args.sender_tokenizer_id or None,
                    receiver_tokenizer_id=args.receiver_tokenizer_id,
                    model_population_id=args.model_population_id,
                )
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
    print(json.dumps({"mode": "executed", "episodes": args.episodes, "output": str(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
