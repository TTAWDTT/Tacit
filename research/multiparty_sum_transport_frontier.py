"""Measure real Tacit envelope costs for optimal finite sum codebooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.multiparty_sum_lossy_frontier import exact_frontier
from tacit import LocalTCPFrameChannel, LocalTCPMessageChannel


TEXT_CHANNEL_ID = "sum-c-0000000000"  # fixed-width identifier sensitivity control
MEDIA_TYPE = "application/octet-stream"
FRAME_ENCODING = "tlu.fixed-width-partition-code-v1"
CODEC_METADATA = {"codec": "tlu.fixed-width-partition-code-v1"}


def _protocol_id(agent_count: int, partitions: list[list[list[int]]]) -> str:
    canonical = json.dumps(
        {"agent_count": agent_count, "partitions": partitions},
        separators=(",", ":"), sort_keys=True,
    ).encode("ascii")
    return "sum-c-" + hashlib.sha256(canonical).hexdigest()[:10]


def _codeword(value: int, partition: list[list[int]]) -> tuple[int, int, str, bytes]:
    try:
        message_id = next(index for index, block in enumerate(partition) if value in block)
    except StopIteration as exc:
        raise ValueError("sender value is outside the codebook partition") from exc
    width = (len(partition) - 1).bit_length()
    text = format(message_id, f"0{width}b") if width else ""
    raw = message_id.to_bytes((width + 7) // 8, "big") if width else b""
    return message_id, width, text, raw


def _decode_text(message: str, width: int, cell_count: int) -> int:
    if len(message) != width or any(bit not in "01" for bit in message):
        raise ValueError("noncanonical text codeword")
    message_id = int(message, 2) if message else 0
    if message_id >= cell_count:
        raise ValueError("unused codeword")
    return message_id


def _decode_frame(payload: bytes, width: int, cell_count: int) -> int:
    expected_size = (width + 7) // 8
    if len(payload) != expected_size:
        raise ValueError("frame payload has a noncanonical byte length")
    message_id = int.from_bytes(payload, "big") if payload else 0
    if width and message_id >= 1 << width:
        raise ValueError("nonzero padding bits or unused codeword")
    if not width and message_id != 0:
        raise ValueError("nonzero padding bits or unused codeword")
    if message_id >= cell_count:
        raise ValueError("unused codeword")
    return message_id


def _simulate_codebook(
    partitions: list[list[list[int]]], decoder_rows: list[dict[str, Any]], *, transport: str,
) -> int:
    decoder = {tuple(row["message_ids"]): row["decoded_sum"] for row in decoder_rows}
    success_count = 0
    # Small finite instances only; the executable oracle restricts m to four.
    from itertools import product
    for inputs in product(range(4), repeat=len(partitions)):
        decoded_ids = []
        for value, partition in zip(inputs, partitions):
            message_id, width, text, raw = _codeword(value, partition)
            recovered = (
                _decode_text(text, width, len(partition))
                if transport == "text"
                else _decode_frame(raw, width, len(partition))
            )
            assert recovered == message_id
            decoded_ids.append(recovered)
        success_count += decoder[tuple(decoded_ids)] == sum(inputs)
    return success_count


def _measure_variant(
    *, agent_count: int, partitions: list[list[list[int]]], protocol_id: str,
    omit_zero_width: bool,
) -> dict[str, Any]:
    text_channel = LocalTCPMessageChannel(lambda _envelope: None)
    frame_channel = LocalTCPFrameChannel(lambda _metadata, _payload: None)
    sender_costs = []
    for index, partition in enumerate(partitions):
        _message_id, width, text, raw = _codeword(0, partition)
        if omit_zero_width and width == 0:
            continue
        sender = f"S{index + 1}"
        text_tx = text_channel.measure(
            text, protocol_id=protocol_id, round_number=index + 1,
            sender=sender, recipient="R",
        )
        frame_tx = frame_channel.measure(
            raw, protocol_id=protocol_id, round_number=index + 1,
            sender=sender, recipient="R", media_type=MEDIA_TYPE,
            encoding=FRAME_ENCODING, payload_metadata=CODEC_METADATA,
        )
        sender_costs.append({
            "sender": sender,
            "source_value_zero_message_id": _message_id,
            "fixed_payload_width_bits": width,
            "text_codeword": text,
            "text_payload_utf8_bytes": len(text.encode("utf-8")),
            "text_payload_serialized_bytes": text_tx.serialized_payload_bytes,
            "text_application_bytes": text_tx.total_application_bytes,
            "text_framing_bytes": text_tx.framing_bytes,
            "frame_payload_hex": raw.hex(),
            "frame_payload_bytes": frame_tx.payload_bytes,
            "frame_application_bytes": frame_tx.total_application_bytes,
            "frame_framing_bytes": frame_tx.framing_bytes,
        })
    return {
        "identifier_policy": "codebook_specific" if protocol_id != TEXT_CHANNEL_ID else "common_equal_length",
        "protocol_id": protocol_id,
        "protocol_id_utf8_bytes": len(protocol_id.encode("utf-8")),
        "sender_messages_including_zero_width_empty_codewords": agent_count,
        "sender_messages_transmitted": len(sender_costs),
        "zero_width_sender_policy": "omit_known_singleton_senders" if omit_zero_width else "send_empty_envelope_for_every_sender",
        "model_calls": 0,
        "hypothetical_llm_role_calls_if_each_active_sender_and_referee_runs_once": len(sender_costs) + 1,
        "text_application_bytes": sum(row["text_application_bytes"] for row in sender_costs),
        "text_payload_utf8_bytes": sum(row["text_payload_utf8_bytes"] for row in sender_costs),
        "text_payload_serialized_bytes": sum(row["text_payload_serialized_bytes"] for row in sender_costs),
        "text_framing_bytes": sum(row["text_framing_bytes"] for row in sender_costs),
        "frame_application_bytes": sum(row["frame_application_bytes"] for row in sender_costs),
        "frame_payload_bytes": sum(row["frame_payload_bytes"] for row in sender_costs),
        "frame_framing_bytes": sum(row["frame_framing_bytes"] for row in sender_costs),
        "per_sender": sender_costs,
    }


def measure_frontier(agent_counts: tuple[int, ...] = (2, 3, 4)) -> dict[str, Any]:
    """Bind exact success frontier points to exact text/frame app-byte costs."""
    counts = {}
    for agent_count in agent_counts:
        budget_rows = []
        for frontier_row in exact_frontier(agent_count):
            partitions = frontier_row["optimal_sender_partitions"]
            simulated_text = _simulate_codebook(partitions, frontier_row["optimal_receiver_decoder"], transport="text")
            simulated_frame = _simulate_codebook(partitions, frontier_row["optimal_receiver_decoder"], transport="frame")
            expected = frontier_row["optimal_success_numerator"]
            if simulated_text != expected or simulated_frame != expected:
                raise ValueError("emitted codebook transport simulation disagrees with exact frontier")
            code_specific_id = _protocol_id(agent_count, partitions)
            budget_rows.append({
                "payload_budget_bits_at_most": frontier_row["payload_budget_bits"],
                "actual_fixed_width_payload_bits": frontier_row["protocol_minimum_bits"],
                "success_numerator": expected,
                "input_vector_count": frontier_row["input_vector_count"],
                "exact_sum_success": frontier_row["optimal_exact_sum_success"],
                "sender_partitions": partitions,
                "success_verified_by_text_codec": simulated_text,
                "success_verified_by_frame_codec": simulated_frame,
                "codebook_setup_and_prompt_cost_included": False,
                "transport": {
                    schedule: {
                        identifier_policy: _measure_variant(
                            agent_count=agent_count, partitions=partitions,
                            protocol_id=identifier, omit_zero_width=(schedule == "omit_zero_width_senders"),
                        )
                        for identifier_policy, identifier in (
                            ("codebook_specific_id", code_specific_id),
                            ("common_equal_length_id", TEXT_CHANNEL_ID),
                        )
                    }
                    for schedule in ("empty_message_for_every_sender", "omit_zero_width_senders")
                },
            })
        counts[str(agent_count)] = budget_rows
    return {
        "schema_version": "tlu.multiparty-sum-transport-frontier.v0.1.0",
        "source_domain": [0, 1, 2, 3],
        "source_prior": "independent_uniform_per_sender",
        "transport_scope": "LocalTCPMessageChannel.measure and LocalTCPFrameChannel.measure exact application-layer bytes; no socket opened",
        "simulation_scope": "all 4^m input vectors exhaustively pass through emitted finite optimal encoder/decoder",
        "accounting": "text codewords are UTF-8 JSON message payloads; frame codewords are opaque byte payloads packed at ceil(width/8) bytes with canonical zero padding; measure both fixed all-sender empty-envelope slots and a schedule that skips known singleton (zero-width) senders",
        "excluded_costs": [
            "codebook/card distribution and setup",
            "sender/receiver prompt and generated tokens",
            "model compute, inference latency, and calls (oracle is model-free)",
            "TCP/IP and link-layer headers",
            "final answer transmission",
        ],
        "identifier_control": "codebook-specific identifier and a fixed equal-length ID are both measured to isolate identifier-size sensitivity",
        "zero_width_schedule_control": "empty frames measure fixed-schedule representation cost; omitting singleton senders exposes the separate message-policy/scheduling contribution",
        "agent_counts": list(agent_counts),
        "source_sha256": {
            "transport_calculator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exact_lossy_frontier_calculator": hashlib.sha256(
                (ROOT / "research" / "multiparty_sum_lossy_frontier.py").read_bytes()
            ).hexdigest(),
            "channel_runtime": hashlib.sha256((ROOT / "tacit" / "channel.py").read_bytes()).hexdigest(),
            "exact_frontier_json": hashlib.sha256(
                (ROOT / "research" / "data" / "MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json").read_bytes()
            ).hexdigest(),
        },
        "frontiers": counts,
    }


if __name__ == "__main__":
    print(json.dumps(measure_frontier(), ensure_ascii=False, indent=2, sort_keys=True))
