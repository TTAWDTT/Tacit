"""Run staged capability controls for the held-out receiver-utility task.

The CLI is dry-run by default. Execution calls only local loopback chat
endpoints and enforces a small per-batch request cap; it never starts a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tacit.channel import LocalTCPMessageChannel
from tacit.runtime import ChatCompletion, ChatModel, OpenAICompatibleClient

from experiments.emergent_ood_v0_2.episodes import generate_ledgers


EXPERIMENT_ID = "emergent-ood-receiver-utility-v0.3-capability"
SCORER_ID = "strict-json-candidate-id-v1"
REGISTERED_CANDIDATE_COUNT = 5
CONDITIONS = ("full_information", "no_message", "natural_language")
CALLS_PER_EPISODE = {"full_information": 1, "no_message": 1, "natural_language": 2}
MAX_MODEL_CALLS_PER_BATCH = 12
REQUEST_TIMEOUT_SECONDS = 30.0
RESOURCE_PREFLIGHT_MAX_AGE_SECONDS = 300
RESOURCE_GATE_LIMITS = {
    "host_cpu_mean_below_percent": 20,
    "host_cpu_each_sample_below_percent": 30,
    "gpu_utilization_below_percent": 25,
    "gpu_memory_below_mib": 1800,
    "free_system_memory_at_least_mib": 6000,
}
SHAPE_LABELS = {0: "circle", 1: "square", 2: "triangle"}
COLOR_LABELS = {0: "red", 1: "green", 2: "blue"}
QUANTITY_LABELS = {0: "one", 1: "two", 2: "three"}
QUANTITY_PHRASES = {0: "once", 1: "twice", 2: "three times"}


def planned_model_calls(episode_count: int, conditions: list[str]) -> int:
    return episode_count * sum(CALLS_PER_EPISODE[name] for name in conditions)


def balanced_block(seed: int, candidate_count: int = 5) -> list[dict[str, Any]]:
    """Select k episodes with one fixed candidate set and every target once."""
    ledgers = generate_ledgers(seed, candidate_count)
    senders = {row["episode_id"]: row for row in ledgers["sender"]}
    receivers = {row["episode_id"]: row for row in ledgers["receiver"]}
    golds = {row["episode_id"]: row for row in ledgers["gold"]}
    grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for episode_id, row in receivers.items():
        key = tuple(sorted(candidate["candidate_id"] for candidate in row["candidates"]))
        grouped[key].append(episode_id)
    candidate_set = min(grouped)
    selected = sorted(grouped[candidate_set], key=lambda episode_id: golds[episode_id]["target_id"])
    if len(selected) != candidate_count:
        raise RuntimeError("selected candidate-set block is not target-balanced")
    if {golds[episode_id]["target_id"] for episode_id in selected} != set(candidate_set):
        raise RuntimeError("selected block must use every candidate once as the target")
    return [
        {
            "episode_seed": seed,
            "sender": senders[episode_id],
            "receiver": receivers[episode_id],
            "gold": golds[episode_id],
        }
        for episode_id in selected
    ]


def _meaning_text(meaning: dict[str, int]) -> str:
    return (
        f"a {SHAPE_LABELS[meaning['shape']]} that is {COLOR_LABELS[meaning['color']]} "
        f"and appears {QUANTITY_PHRASES[meaning['quantity']]}"
    )


def _candidate_text(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "shape": SHAPE_LABELS[candidate["shape"]],
        "color": COLOR_LABELS[candidate["color"]],
        "quantity": QUANTITY_LABELS[candidate["quantity"]],
    }


def _receiver_messages(receiver_view: dict[str, Any], clue: str | None) -> list[dict[str, str]]:
    system = (
        "You are the receiver in a private-target matching task. Compare the clue with the candidates. "
        "Return exactly one JSON object with the single key candidate_id and its matching ID as a string. "
        "Do not add explanation or other keys."
    )
    user = {
        "task": "Select the one candidate whose shape, color, and quantity all match the clue.",
        "clue": clue,
        "candidates": [_candidate_text(row) for row in receiver_view["candidates"]],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
    ]


def _parse_choice(text: str, candidate_ids: list[str]) -> tuple[str | None, bool]:
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(decoded, dict) or set(decoded) != {"candidate_id"}:
        return None, False
    choice = decoded["candidate_id"]
    if not isinstance(choice, str) or choice not in candidate_ids:
        return None, False
    return choice, True


def validate_capability_ledger(
    path: Path | None,
    episodes: list[dict[str, Any]],
    *,
    receiver_model: str,
    receiver_tokenizer_id: str,
    model_population_id: str,
) -> None:
    """Require a perfect, same-receiver full-information ledger before follow-ups."""
    if path is None:
        raise ValueError("later stages require --capability-ledger from the full_information stage")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read capability ledger: {path}") from exc
    expected = {episode["sender"]["episode_id"]: episode for episode in episodes}
    indexed = {row.get("episode_id"): row for row in rows if isinstance(row, dict)}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError("capability ledger must contain exactly the five current paired episode IDs")
    for episode_id, episode in expected.items():
        row = indexed[episode_id]
        diagnostics = row.get("diagnostics", {})
        calls = row.get("model_calls", [])
        receiver_calls = [call for call in calls if isinstance(call, dict) and call.get("agent") == "receiver"]
        if row.get("protocol", {}).get("policy_id") != "full_information_capability_control":
            raise ValueError("capability ledger contains a non-full-information condition")
        if row.get("outcome", {}).get("joint_success") is not True or diagnostics.get("answer_format_valid") is not True:
            raise ValueError("all five full-information episodes must be exact and format-valid")
        if diagnostics.get("condition") != "full_information":
            raise ValueError("capability ledger condition labels are invalid")
        if diagnostics.get("sender_target_id") != episode["sender"]["private_target_id"]:
            raise ValueError("capability ledger target does not match this episode")
        expected_candidates = [row["candidate_id"] for row in episode["receiver"]["candidates"]]
        if diagnostics.get("candidate_ids_in_receiver_order") != expected_candidates:
            raise ValueError("capability ledger candidate view does not match this episode")
        if len(receiver_calls) != 1 or receiver_calls[0].get("stage") != "full_information_answer":
            raise ValueError("each capability episode must contain exactly one full-information receiver call")
        if receiver_calls[0].get("tokenizer") != receiver_tokenizer_id:
            raise ValueError("receiver tokenizer differs from the successful capability screen")
        if receiver_calls[0].get("model") != receiver_model:
            raise ValueError("receiver call model differs from the successful capability screen")
        stratum = row.get("stratum", {})
        if stratum.get("model_population_id") != model_population_id:
            raise ValueError("receiver model population differs from the successful capability screen")
        if stratum.get("agent_models", {}).get("receiver") != receiver_model:
            raise ValueError("receiver model differs from the successful capability screen")


def validate_resource_preflight(
    path: Path | None,
    *,
    required_ports: set[int],
    machine_name: str | None = None,
    now: datetime | None = None,
) -> None:
    """Require a recent passing read-only preflight for this host and endpoint ports."""
    if path is None:
        raise ValueError("--execute requires --resource-preflight from resource_preflight.ps1")
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read resource preflight report: {path}") from exc
    if not isinstance(report, dict) or report.get("schema") != "tlu.local_resource_preflight.v1":
        raise ValueError("resource preflight report schema is invalid")
    if report.get("status") != "eligible":
        raise ValueError("resource preflight did not pass")
    try:
        sampled_at = datetime.fromisoformat(report["sampled_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resource preflight timestamp is invalid") from exc
    if sampled_at.tzinfo is None:
        raise ValueError("resource preflight timestamp must include a timezone")
    current_time = now or datetime.now(timezone.utc)
    age = (current_time - sampled_at.astimezone(timezone.utc)).total_seconds()
    if age < -30 or age > RESOURCE_PREFLIGHT_MAX_AGE_SECONDS:
        raise ValueError("resource preflight must be no more than five minutes old")
    expected_machine = machine_name if machine_name is not None else os.environ.get("COMPUTERNAME")
    if not expected_machine:
        raise ValueError("cannot verify resource preflight host identity")
    if report.get("machine_name") != expected_machine:
        raise ValueError("resource preflight belongs to a different machine")
    if report.get("limits") != RESOURCE_GATE_LIMITS:
        raise ValueError("resource preflight limits do not match the frozen gate")
    ports = report.get("requested_ports")
    if not isinstance(ports, list) or any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError("resource preflight requested_ports are invalid")
    if len(set(ports)) != len(ports):
        raise ValueError("resource preflight requested_ports must be unique")
    if not required_ports.issubset(set(ports)):
        raise ValueError("resource preflight did not check every configured endpoint port")
    if report.get("listening_requested_ports") != []:
        raise ValueError("a configured endpoint port was already in use during preflight")
    if any(report.get(key) is not False for key in (
        "model_artifact_hashed", "model_artifact_read", "model_loaded", "service_started",
    )) or report.get("inference_requests") != 0:
        raise ValueError("resource preflight must be read-only and precede model/service startup")
    observed = report.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("resource preflight observed measurements are missing")
    samples = observed.get("host_cpu_samples_percent")
    if not isinstance(samples, list) or len(samples) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in samples
    ):
        raise ValueError("resource preflight must contain three finite CPU samples")
    mean_cpu = sum(samples) / len(samples)
    max_cpu = max(samples)
    reported_mean = observed.get("host_cpu_mean_percent")
    reported_max = observed.get("host_cpu_max_percent")
    numeric_values = (
        reported_mean, reported_max,
        observed.get("gpu_utilization_percent"),
        observed.get("gpu_memory_used_mib"),
        observed.get("gpu_memory_total_mib"),
        observed.get("free_system_memory_mib"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in numeric_values
    ):
        raise ValueError("resource preflight GPU/memory measurements are missing or invalid")
    if abs(reported_mean - mean_cpu) > 0.02:
        raise ValueError("resource preflight CPU mean does not match its samples")
    if abs(reported_max - max_cpu) > 0.02:
        raise ValueError("resource preflight CPU maximum does not match its samples")
    if any(value < 0 or value > 100 for value in (*samples, reported_mean, reported_max, observed["gpu_utilization_percent"])):
        raise ValueError("resource preflight utilization percentages are outside 0..100")
    if (
        observed["gpu_memory_used_mib"] < 0
        or observed["gpu_memory_total_mib"] <= 0
        or observed["gpu_memory_used_mib"] > observed["gpu_memory_total_mib"]
        or observed["free_system_memory_mib"] < 0
    ):
        raise ValueError("resource preflight memory measurements are outside valid ranges")
    if mean_cpu >= 20 or max_cpu >= 30:
        raise ValueError("resource preflight CPU measurements exceed the frozen gate")
    if observed["gpu_utilization_percent"] >= 25 or observed["gpu_memory_used_mib"] >= 1800:
        raise ValueError("resource preflight GPU measurements exceed the frozen gate")
    if observed["free_system_memory_mib"] < 6000:
        raise ValueError("resource preflight free memory is below the frozen gate")


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


def run_condition(
    *,
    episode: dict[str, Any],
    condition: str,
    sender_model: ChatModel | None,
    receiver_model: ChatModel,
    sender_tokenizer_id: str | None,
    receiver_tokenizer_id: str,
    model_population_id: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    sender_view = episode["sender"]
    receiver_view = episode["receiver"]
    gold = episode["gold"]
    candidate_ids = [row["candidate_id"] for row in receiver_view["candidates"]]
    calls: list[dict[str, Any]] = []
    received_text: list[str] = []
    message: str | None = None
    wall_started = time.perf_counter()

    if condition == "full_information":
        clue = _meaning_text(sender_view["private_target"])
        receiver_result = receiver_model.complete(_receiver_messages(receiver_view, clue))
        calls.append(_call_record("receiver", "full_information_answer", receiver_result, receiver_tokenizer_id))
        answer_text = receiver_result.text
        policy_id, code_id, decoder_id = "full_information_capability_control", "oracle-clue-v1", SCORER_ID
        transmissions: list[dict[str, Any]] = []
    elif condition == "no_message":
        receiver_result = receiver_model.complete(_receiver_messages(receiver_view, None))
        calls.append(_call_record("receiver", "no_message_answer", receiver_result, receiver_tokenizer_id))
        answer_text = receiver_result.text
        policy_id, code_id, decoder_id = "no_message", "none", SCORER_ID
        transmissions = []
    else:
        if sender_model is None or not sender_tokenizer_id:
            raise ValueError("natural_language requires sender model and tokenizer IDs")
        sender_prompt = [
            {
                "role": "system",
                "content": "Describe the target in one concise, ordinary English sentence. State its shape, color, and quantity. Do not mention candidate IDs or add any other facts.",
            },
            {
                "role": "user",
                "content": json.dumps({
                    "private_target": _candidate_text({
                        "candidate_id": sender_view["private_target_id"],
                        **sender_view["private_target"],
                    })
                }, ensure_ascii=False, sort_keys=True),
            },
        ]
        sender_result = sender_model.complete(sender_prompt)
        calls.append(_call_record("sender", "natural_language_encode", sender_result, sender_tokenizer_id))
        message = sender_result.text

        def receive(envelope: dict[str, Any]) -> None:
            received_text.append(envelope["payload"])

        with LocalTCPMessageChannel(receive) as channel:
            transmission = channel.send(
                message,
                protocol_id="plain-english-description-v0.3",
                round_number=1,
                sender="sender",
                recipient="receiver",
            )
        if received_text != [message]:
            raise RuntimeError("receiver did not receive exactly the sender's verbatim message")
        receiver_result = receiver_model.complete(_receiver_messages(receiver_view, received_text[0]))
        calls.append(_call_record("receiver", "natural_language_decode", receiver_result, receiver_tokenizer_id))
        answer_text = receiver_result.text
        policy_id, code_id, decoder_id = "natural_language_baseline", "freeform-english-v1", SCORER_ID
        transmissions = [transmission.cost_record()]

    answer_id, parse_valid = _parse_choice(answer_text, candidate_ids)
    success = answer_id == gold["target_id"]
    episode_id = sender_view["episode_id"]
    block_id = "|".join(sorted(candidate_ids))
    return {
        "schema_version": "tlu.costs.v3",
        "episode_id": episode_id,
        "stratum": {
            "experiment_id": EXPERIMENT_ID,
            "task_id": "three-attribute-heldout-reference-v1",
            "split": "heldout-composition",
            "task_parameters": {
                "candidate_count": len(candidate_ids),
                "target_count": 9,
                "balanced_block_id": block_id,
            },
            "model_population_id": model_population_id,
            "agent_models": {
                "sender": getattr(sender_model, "model_name", "absent") if sender_model is not None else "absent",
                "receiver": getattr(receiver_model, "model_name", "unknown"),
            },
            "scorer_id": SCORER_ID,
        },
        "protocol": {"policy_id": policy_id, "code_id": code_id, "decoder_id": decoder_id},
        "outcome": {"joint_success": success, "answer_score": 1.0 if success else 0.0},
        "diagnostics": {
            "condition": condition,
            "generation_seed": episode["episode_seed"],
            "sender_target_id": sender_view["private_target_id"],
            "candidate_ids_in_receiver_order": candidate_ids,
            "target_position": gold["target_position"],
            "message_text": message,
            "answer_text": answer_text,
            "answer_candidate_id": answer_id,
            "answer_format_valid": parse_valid,
            "sender_truncated": calls[0]["truncated"] if condition == "natural_language" else None,
            "receiver_truncated": calls[-1]["truncated"],
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


def _endpoint_port(value: str) -> int:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use HTTP or HTTPS")
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="call the configured local loopback endpoint")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["full_information"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path(".cache/emergent_ood_v0_3/full_information.jsonl"))
    parser.add_argument("--capability-ledger", type=Path, help="perfect full_information JSONL required before executing later conditions")
    parser.add_argument("--resource-preflight", type=Path, help="passing report from resource_preflight.ps1, at most five minutes old")
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
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    episodes = balanced_block(args.seed, REGISTERED_CANDIDATE_COUNT)
    calls_planned = planned_model_calls(len(episodes), args.conditions)
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "conditions": args.conditions,
            "episodes_per_condition": len(episodes),
            "paired_episode_ids": [row["sender"]["episode_id"] for row in episodes],
            "candidate_count": REGISTERED_CANDIDATE_COUNT,
            "no_message_bayes_accuracy": 1 / REGISTERED_CANDIDATE_COUNT,
            "fixed_width_zero_error_payload_floor_bits": 4,
            "planned_model_calls": calls_planned,
            "maximum_model_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
            "fits_call_cap": calls_planned <= MAX_MODEL_CALLS_PER_BATCH,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "output": str(args.output),
            "inference_started": False,
        }, indent=2))
        return 0
    if calls_planned > MAX_MODEL_CALLS_PER_BATCH:
        parser.error(f"batch plans {calls_planned} requests; hard cap is {MAX_MODEL_CALLS_PER_BATCH}")
    if not args.receiver_model or not args.receiver_tokenizer_id:
        parser.error("--execute requires receiver model and tokenizer IDs")
    if any(condition != "full_information" for condition in args.conditions):
        try:
            validate_capability_ledger(
                args.capability_ledger,
                episodes,
                receiver_model=args.receiver_model,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                model_population_id=args.model_population_id,
            )
        except ValueError as exc:
            parser.error(str(exc))
    needs_sender = "natural_language" in args.conditions
    if needs_sender and (not args.sender_model or not args.sender_tokenizer_id):
        parser.error("natural_language requires sender model and tokenizer IDs")
    try:
        sender_endpoint = _loopback_url(args.sender_base_url or args.base_url) if needs_sender else None
        receiver_endpoint = _loopback_url(args.receiver_base_url or args.base_url)
        required_ports = {_endpoint_port(receiver_endpoint)}
        if sender_endpoint:
            required_ports.add(_endpoint_port(sender_endpoint))
        validate_resource_preflight(args.resource_preflight, required_ports=required_ports)
    except ValueError as exc:
        parser.error(str(exc))
    target = args.output.resolve()
    if target.exists() and not args.force:
        parser.error(f"refusing to overwrite {target}; pass --force explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    sender = _NamedClient(OpenAICompatibleClient(
        sender_endpoint or "", args.sender_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=128, follow_redirects=False,
    )) if needs_sender else None
    receiver = _NamedClient(OpenAICompatibleClient(
        receiver_endpoint, args.receiver_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=64, follow_redirects=False,
    ))
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for episode in episodes:
            for condition in args.conditions:
                row = run_condition(
                    episode=episode,
                    condition=condition,
                    sender_model=sender,
                    receiver_model=receiver,
                    sender_tokenizer_id=args.sender_tokenizer_id or None,
                    receiver_tokenizer_id=args.receiver_tokenizer_id,
                    model_population_id=args.model_population_id,
                )
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
    print(json.dumps({
        "mode": "executed", "conditions": args.conditions,
        "episodes_per_condition": len(episodes), "model_calls": calls_planned,
        "output": str(target),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
