"""Demonstrate a hidden-input multi-sender task through the Tacit runtime.

No endpoint is contacted until ``main`` runs. This small example is an
operational illustration, not a benchmark or a language-superiority result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tacit import DialogueProtocolCard, OpenAICompatibleClient, exchange_dialogue
from experiments.emergent_ood_v0_3.runner import validate_resource_preflight


RECEIVER = "R"
MAX_REQUESTS_PER_EPISODE = 12
MAX_WIRE_BUDGET_BYTES = 4096


def run_sum_episode(
    values: list[int],
    *,
    sender_clients: list[OpenAICompatibleClient],
    receiver_client: OpenAICompatibleClient,
    condition: str = "communicate",
    wire_budget_bytes: int = 4096,
    request_cap: int = 12,
) -> dict[str, object]:
    """Run one strictly scored private-sum episode with supplied chat clients."""
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("values must contain at least two private sender values")
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in values):
        raise ValueError("each private value must be an integer in {0, 1, 2, 3}")
    if not isinstance(sender_clients, list) or len(sender_clients) != len(values):
        raise ValueError("sender_clients must contain one client per private value")
    if condition not in {"communicate", "no_message", "full_information"}:
        raise ValueError("condition must be communicate, no_message, or full_information")
    if isinstance(request_cap, bool) or not isinstance(request_cap, int) or request_cap < 1:
        raise ValueError("request_cap must be a positive integer")
    if request_cap > MAX_REQUESTS_PER_EPISODE:
        raise ValueError(f"request_cap cannot exceed the frozen {MAX_REQUESTS_PER_EPISODE}-call ceiling")
    if (
        isinstance(wire_budget_bytes, bool)
        or not isinstance(wire_budget_bytes, int)
        or not 0 <= wire_budget_bytes <= MAX_WIRE_BUDGET_BYTES
    ):
        raise ValueError(f"wire_budget_bytes must be between zero and {MAX_WIRE_BUDGET_BYTES}")
    planned_calls = len(values) + 1 if condition == "communicate" else 1
    if planned_calls > request_cap:
        raise ValueError(f"condition requires {planned_calls} calls, exceeding request cap {request_cap}")
    if len(values) > 255:
        raise ValueError("the dialogue protocol card supports at most 255 senders plus the receiver")
    sender_names = [f"S{index + 1}" for index in range(len(values))]
    agents = {name: client for name, client in zip(sender_names, sender_clients)}
    agents[RECEIVER] = receiver_client
    instructions = {
        name: "When scheduled, send exactly your own private integer as a base-10 numeral. Do not add a label or explanation."
        for name in sender_names
    }
    instructions[RECEIVER] = (
        "Use only the private context and messages visible to you. Add the integer values "
        "from all sender messages and return only the exact sum as a base-10 integer."
    )
    protocol = DialogueProtocolCard("private-sum-decimal-v0", instructions)
    task = (
        f"Compute the exact sum of the private integers held by senders {', '.join(sender_names)}. "
        "Use only information visible to you and return one base-10 integer."
    )
    contexts = {name: f"Your private integer is {value}." for name, value in zip(sender_names, values)}
    contexts[RECEIVER] = ""
    if condition == "full_information":
        contexts[RECEIVER] = "All private integers, in sender order: " + ", ".join(map(str, values))
    schedule = tuple((name, RECEIVER) for name in sender_names) if condition == "communicate" else ()
    result = exchange_dialogue(
        agents,
        protocol=protocol,
        private_contexts=contexts,
        schedule=schedule,
        task=task,
        max_turns=len(sender_names),
        wire_budget_bytes=wire_budget_bytes,
        final_answer_agent=RECEIVER,
        final_answer_instruction="Return only one non-negative base-10 integer.",
    )
    output = "" if result.final_submission is None else result.final_submission.text.strip()
    match = re.fullmatch(r"(?:0|[1-9][0-9]*)", output)
    prediction = None if match is None else int(output)
    return {
        "protocol_id": result.protocol_id,
        "condition": condition,
        "sender_count": len(sender_names),
        "expected_sum": sum(values),
        "prediction": prediction,
        "exact_success": prediction == sum(values),
        "model_calls": result.model_calls,
        "wire_bytes": result.wire_bytes,
        "wire_budget_bytes": result.wire_budget_bytes,
        "stop_reason": result.stop_reason,
        "message_transmissions": result.transmission_records(),
        "model_call_records": result.model_call_records(),
        "raw_answer": output,
    }


def _client(base_url: str, model: str, api_key: str | None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=30,
        max_tokens=96,
        follow_redirects=False,
        temperature=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, nargs="+", required=True, help="private values, each 0 through 3")
    parser.add_argument(
        "--condition", choices=("communicate", "no_message", "full_information"),
        default="communicate",
    )
    parser.add_argument("--wire-budget-bytes", type=int, default=4096)
    parser.add_argument("--request-cap", type=int, default=12, help="hard planned-call ceiling for this episode")
    parser.add_argument(
        "--resource-preflight", type=Path, required=True,
        help="fresh passing local resource report covering sender and receiver ports",
    )
    args = parser.parse_args()
    sender_url = os.environ.get("TLU_SUM_SENDER_URL", "http://127.0.0.1:8000/v1")
    receiver_url = os.environ.get("TLU_SUM_RECEIVER_URL", "http://127.0.0.1:8001/v1")
    ports: set[int] = set()
    try:
        for endpoint in (sender_url, receiver_url):
            parsed = urlsplit(endpoint)
            if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                parser.error("the example only permits HTTP loopback endpoints")
            ports.add(parsed.port or 80)
    except ValueError as exc:
        parser.error(f"invalid loopback endpoint URL: {exc}")
    try:
        validate_resource_preflight(args.resource_preflight, required_ports=ports)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sender_model = os.environ.get("TLU_SUM_SENDER_MODEL", "local-sender")
    receiver_model = os.environ.get("TLU_SUM_RECEIVER_MODEL", "local-receiver")
    sender_clients = [
        _client(sender_url, sender_model, os.environ.get("TLU_SUM_SENDER_API_KEY"))
        for _ in args.values
    ]
    receiver_client = _client(receiver_url, receiver_model, os.environ.get("TLU_SUM_RECEIVER_API_KEY"))
    report = run_sum_episode(
        args.values,
        sender_clients=sender_clients,
        receiver_client=receiver_client,
        condition=args.condition,
        wire_budget_bytes=args.wire_budget_bytes,
        request_cap=args.request_cap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
