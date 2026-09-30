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
MESSAGE_FORMATS = ("decimal", "json", "labeled", "binary", "sentence")
SUM_TASK = (
    "Team objective: find the exact sum of the sender-held private integers. "
    "Assume the private integers are drawn independently and uniformly from {0, 1, 2, 3}. "
    "Use only information visible to your role and follow your role instruction."
)


def _format_instructions(message_format: str) -> tuple[str, str]:
    if message_format == "decimal":
        return (
            "Transmit exactly one base-10 digit from 0 through 3 for your private integer.",
            "Each sender message is one base-10 digit from 0 through 3. Parse each delivered message and sum the values.",
        )
    if message_format == "json":
        return (
            'Transmit exactly one JSON object with one integer field, e.g. {"value": 2}.',
            'Parse each message as a JSON object with exactly the integer field "value" and sum the field values.',
        )
    if message_format == "labeled":
        return (
            "Transmit exactly v=N, where N is your private integer from 0 through 3.",
            "Parse each sender message using exactly v=N, where N is one digit from 0 through 3, then sum N.",
        )
    if message_format == "binary":
        return (
            "Use this shared base-2 text codebook: 00 means 0, 01 means 1, 10 means 2, and 11 means 3. Transmit exactly the matching two-character string.",
            "Decode each exact two-character text message with 00→0, 01→1, 10→2, 11→3, then sum the decoded values.",
        )
    if message_format == "sentence":
        return (
            "Transmit exactly one sentence in this template: My private integer is N. Replace N with your digit from 0 through 3.",
            "Parse each message using exactly the sentence template My private integer is N. where N is one digit from 0 through 3, then sum N.",
        )
    raise ValueError(f"unsupported message_format: {message_format}")


def _role_instructions(message_format: str, sender_names: list[str]) -> dict[str, str]:
    """Build the exact system instructions used by every private-sum episode."""
    sender_instruction, receiver_format_instruction = _format_instructions(message_format)
    instructions = {name: sender_instruction for name in sender_names}
    instructions[RECEIVER] = (
        "Use only information visible in your public/private context and received sender messages. "
        f"{receiver_format_instruction} If a full list of private values is directly present in your private context, "
        "sum those values instead. The public context gives sender_count=m. If fewer than m distinct sender inputs "
        "are known because communication is disabled, a message is missing, or a message is malformed, let k be the "
        "number of unknown inputs. Under the task's independent uniform prior on {0, 1, 2, 3}, use floor(3k/2) "
        "as one optimal exact-sum guess for those inputs, then add all known values. Return only the exact sum as a "
        "base-10 integer."
    )
    return instructions


def _decode_sender_message(message_format: str, message: str) -> int | None:
    patterns = {
        "decimal": r"[0-3]",
        "labeled": r"v=([0-3])",
        "sentence": r"My private integer is ([0-3])\.",
    }
    if message_format == "binary":
        return {"00": 0, "01": 1, "10": 2, "11": 3}.get(message)
    if message_format == "json":
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON field")
                result[key] = value
            return result

        try:
            data = json.loads(message, object_pairs_hook=unique_pairs)
        except (json.JSONDecodeError, ValueError):
            return None
        if (
            isinstance(data, dict)
            and set(data) == {"value"}
            and isinstance(data["value"], int)
            and not isinstance(data["value"], bool)
            and data["value"] in range(4)
        ):
            return data["value"]
        return None
    pattern = patterns.get(message_format)
    if pattern is None:
        raise ValueError(f"unsupported message_format: {message_format}")
    match = re.fullmatch(pattern, message)
    if match is None:
        return None
    return int(match.group(1) if match.lastindex else match.group(0))


def run_sum_episode(
    values: list[int],
    *,
    sender_clients: list[OpenAICompatibleClient],
    receiver_client: OpenAICompatibleClient,
    condition: str = "communicate",
    message_format: str = "decimal",
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
    if message_format not in MESSAGE_FORMATS:
        raise ValueError(f"message_format must be one of {', '.join(MESSAGE_FORMATS)}")
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
    instructions = _role_instructions(message_format, sender_names)
    protocol = DialogueProtocolCard(f"private-sum-{message_format}-v0", instructions)
    contexts = {
        name: f"Public metadata: sender_count={len(sender_names)}. Your private integer is {value}."
        for name, value in zip(sender_names, values)
    }
    contexts[RECEIVER] = f"Public metadata: sender_count={len(sender_names)}."
    if condition == "full_information":
        contexts[RECEIVER] += " All private integers, in sender order: " + ", ".join(map(str, values))
    schedule = tuple((name, RECEIVER) for name in sender_names) if condition == "communicate" else ()
    result = exchange_dialogue(
        agents,
        protocol=protocol,
        private_contexts=contexts,
        schedule=schedule,
        task=SUM_TASK,
        max_turns=len(sender_names),
        wire_budget_bytes=wire_budget_bytes,
        final_answer_agent=RECEIVER,
        final_answer_instruction="Return only one non-negative base-10 integer.",
    )
    output = "" if result.final_submission is None else result.final_submission.text.strip()
    match = re.fullmatch(r"(?:0|[1-9][0-9]*)", output)
    prediction = None if match is None else int(output)
    expected_by_sender = dict(zip(sender_names, values))
    sender_messages = []
    for turn in result.turns:
        parsed_value = _decode_sender_message(message_format, turn.completion.text)
        expected_value = expected_by_sender[turn.speaker]
        sender_messages.append({
            "sender": turn.speaker,
            "raw_message": turn.completion.text,
            "syntax_valid": parsed_value is not None,
            "parsed_value": parsed_value,
            "expected_value_for_scorer": expected_value,
            "value_faithful": parsed_value == expected_value,
            "delivered": turn.transmission is not None,
        })
    syntax_valid_count = sum(row["syntax_valid"] is True for row in sender_messages)
    faithful_count = sum(row["value_faithful"] is True for row in sender_messages)
    delivered_messages = [row for row in sender_messages if row["delivered"] is True]
    delivered_syntax_valid_count = sum(row["syntax_valid"] is True for row in delivered_messages)
    delivered_faithful_count = sum(row["value_faithful"] is True for row in delivered_messages)
    return {
        "protocol_id": result.protocol_id,
        "condition": condition,
        "message_format": message_format,
        "sender_count": len(sender_names),
        "assumed_input_prior": "independent_uniform_integer_0_to_3",
        "expected_sum": sum(values),
        "prediction": prediction,
        "exact_success": prediction == sum(values),
        "sender_message_count": len(sender_messages),
        "sender_generated_message_count": len(sender_messages),
        "sender_delivered_message_count": len(delivered_messages),
        "sender_syntax_valid_count": syntax_valid_count,
        "sender_value_faithful_count": faithful_count,
        "all_sender_messages_syntax_valid": None if not sender_messages else syntax_valid_count == len(sender_messages),
        "all_sender_values_faithful": None if not sender_messages else faithful_count == len(sender_messages),
        "sender_delivered_syntax_valid_count": delivered_syntax_valid_count,
        "sender_delivered_value_faithful_count": delivered_faithful_count,
        "all_delivered_sender_messages_syntax_valid": (
            None if not delivered_messages else delivered_syntax_valid_count == len(delivered_messages)
        ),
        "all_delivered_sender_values_faithful": (
            None if not delivered_messages else delivered_faithful_count == len(delivered_messages)
        ),
        "sender_messages": sender_messages,
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
    parser.add_argument("--message-format", choices=MESSAGE_FORMATS, default="decimal")
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
        message_format=args.message_format,
        wire_budget_bytes=args.wire_budget_bytes,
        request_cap=args.request_cap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
