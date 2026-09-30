"""Exact, model-free byte accounting for the multiparty private-sum formats.

This measures the current text channel's application serialization only. It
does not call ``send``, open a socket, tokenize prompts, or estimate task
success. The committed JSON is a reproducible transport-cost diagnostic, not a
protocol-efficiency result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacit import LocalTCPMessageChannel
from examples.multiparty_private_sum import MESSAGE_FORMATS


EXAMPLE_SOURCE = ROOT / "examples" / "multiparty_private_sum.py"
DEFAULT_OUTPUT = ROOT / "research" / "data" / "MULTIPARTY_SUM_WIRE_ACCOUNTING_V0_1.json"
AGENT_COUNTS = tuple(range(2, 12))  # twelve-call cap: m sender calls + one receiver call


def reference_message(message_format: str, value: int) -> str:
    """Return one canonical valid example message for a four-value domain."""
    if value not in range(4):
        raise ValueError("value must be in {0, 1, 2, 3}")
    if message_format == "decimal":
        return str(value)
    if message_format == "json":
        return json.dumps({"value": value}, separators=(",", ":"))
    if message_format == "labeled":
        return f"v={value}"
    if message_format == "binary":
        return f"{value:02b}"
    if message_format == "sentence":
        return f"My private integer is {value}."
    raise ValueError(f"unknown message format: {message_format}")


def measure() -> dict[str, object]:
    channel = LocalTCPMessageChannel(lambda _envelope: None)
    format_rows = []
    for message_format in MESSAGE_FORMATS:
        scenario_rows = []
        protocol_scenarios = (
            ("shipped_format_specific_id", f"private-sum-{message_format}-v0"),
            ("shared_equal_length_id", "private-sum-common-v0"),
        )
        for id_policy, protocol_id in protocol_scenarios:
            count_rows = []
            for sender_count in AGENT_COUNTS:
                costs = []
                for index in range(sender_count):
                    sender = f"S{index + 1}"
                    value = index % 4
                    message = reference_message(message_format, value)
                    transmission = channel.measure(
                        message,
                        protocol_id=protocol_id,
                        round_number=index + 1,
                        sender=sender,
                        recipient="R",
                    )
                    costs.append({
                        "sender": sender,
                        "value": value,
                        "message": message,
                        "logical_payload_utf8_bytes": transmission.logical_payload_bytes,
                        "serialized_payload_bytes": transmission.serialized_payload_bytes,
                        "framing_bytes": transmission.framing_bytes,
                        "total_application_bytes": transmission.total_application_bytes,
                    })
                total_bytes = sum(row["total_application_bytes"] for row in costs)
                count_rows.append({
                    "sender_count": sender_count,
                    "scheduled_messages": sender_count,
                    "model_calls_if_one_call_per_sender_and_receiver": sender_count + 1,
                    "ideal_zero_error_payload_floor_bits": 2 * sender_count,
                    "logical_payload_utf8_bytes": sum(row["logical_payload_utf8_bytes"] for row in costs),
                    "serialized_payload_bytes": sum(row["serialized_payload_bytes"] for row in costs),
                    "framing_and_acknowledgment_bytes": sum(row["framing_bytes"] for row in costs),
                    "total_application_bytes": total_bytes,
                    "application_wire_bits_over_ideal_payload_bits": 8 * total_bytes / (2 * sender_count),
                    "per_message": costs,
                })
            scenario_rows.append({
                "identifier_policy": id_policy,
                "protocol_id": protocol_id,
                "counts": count_rows,
            })
        format_rows.append({
            "message_format": message_format,
            "scenarios": scenario_rows,
        })

    return {
        "schema": "tlu.multiparty_sum_wire_accounting.v0.1",
        "scope": "exact application-byte serialization diagnostic; not an LLM protocol comparison",
        "channel": "LocalTCPMessageChannel.measure; length-prefixed UTF-8 JSON text envelope plus one-byte delivery acknowledgment",
        "channel_boundary": "application-layer loopback bytes; TCP/IP/link-layer headers excluded",
        "task": "M=4 private-sum calibration; sender values cycle 0,1,2,3 by sender order",
        "identifier_sensitivity": "report both the example's format-specific protocol IDs and one shared equal-length protocol ID; only the latter isolates message representation from repeated protocol-ID length",
        "frozen_call_ceiling": 12,
        "agent_counts": list(AGENT_COUNTS),
        "format_source_sha256": hashlib.sha256(EXAMPLE_SOURCE.read_bytes()).hexdigest(),
        "excluded_costs": [
            "sender and receiver system-instruction tokens",
            "complete prompt and generation tokens",
            "model compute, latency, retries, and task success",
            "protocol-card/codebook distribution or setup",
            "final answer bytes (the SDK seals the answer outside the inter-agent channel)",
        ],
        "formats": format_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true", help="print JSON instead of writing a file")
    args = parser.parse_args()
    result = measure()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
