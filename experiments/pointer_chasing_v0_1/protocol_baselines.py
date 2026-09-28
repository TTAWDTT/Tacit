"""Exact model-free communication controls for pointer chasing.

The byte counts use unframed ASCII payloads. Transport envelopes, tokenizer
costs, and inference cost are intentionally outside this analytic ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_tasks import decode_pointer, encode_pointer, pointer_trace, validate_episode


def encode_function(values: list[int], size: int) -> str:
    """Serialize a complete function as concatenated fixed-width binary offsets."""
    if not isinstance(values, list) or len(values) != size:
        raise ValueError("function must contain exactly n values")
    return "".join(encode_pointer(value, size) for value in values)


def decode_function(payload: str, size: int) -> list[int]:
    """Decode a complete function from fixed-width binary offsets."""
    if not isinstance(payload, str):
        raise ValueError("function payload must be text")
    width = (size - 1).bit_length()
    if len(payload) != size * width:
        raise ValueError("function payload has the wrong fixed width")
    return [decode_pointer(payload[i:i + width], size) for i in range(0, len(payload), width)]


def full_map_exchange(episode: dict) -> dict:
    """Both agents reveal their complete map in one simultaneous batch.

    After receiving the other map, both agents know the complete instance and
    independently compute the same answer. This is a deliberately simple exact
    bilateral control, not an optimized low-round protocol.
    """
    validate_episode(episode)
    size = episode["size"]
    width = (size - 1).bit_length()
    payload_a = encode_function(episode["agent_a_view"]["function_values_1_based"], size)
    payload_b = encode_function(episode["agent_b_view"]["function_values_1_based"], size)
    # Each endpoint computes from its own map and the map it received.
    function_a = episode["agent_a_view"]["function_values_1_based"]
    function_b = episode["agent_b_view"]["function_values_1_based"]
    received_b_at_a = decode_function(payload_b, size)
    received_a_at_b = decode_function(payload_a, size)
    answer_a = pointer_trace(function_a, received_b_at_a, episode["depth"])[-1] % 2
    answer_b = pointer_trace(received_a_at_b, function_b, episode["depth"])[-1] % 2
    if answer_a != answer_b:
        raise AssertionError("agents computed different answers after exchanging full maps")
    messages = [
        {"sender": "agent_a", "receiver": "agent_b", "payload_bits": payload_a},
        {"sender": "agent_b", "receiver": "agent_a", "payload_bits": payload_b},
    ]
    payload_bits = sum(len(message["payload_bits"]) for message in messages)
    return {
        "policy_id": "full-map-simultaneous-v1",
        "answer_bit": answer_a,
        "agent_answers": {"agent_a": answer_a, "agent_b": answer_b},
        "both_agents_can_compute_answer": True,
        "synchronous_batches": 1,
        "directed_transmissions": len(messages),
        "bits_per_function_value": width,
        "messages": messages,
        "aggregate_payload_bits": payload_bits,
        "aggregate_payload_bytes_ascii": payload_bits,
        "framing_bytes_included": False,
        "transport_envelope_included": False,
        "recipient_tokenizer_tokens": None,
        "inference_cost": None,
    }


def oracle_frontier_point(size: int, depth: int) -> dict:
    """Return exact analytic costs for two zero-error oracle controls."""
    if type(size) is not int or size < 2 or size % 2:
        raise ValueError("size must be an even integer >= 2")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be positive")
    width = (size - 1).bit_length()
    relay_bits = depth * width
    full_map_bits = 2 * size * width
    return {
        "size": size,
        "depth": depth,
        "zero_error": True,
        "both_agents_independently_compute_answer": True,
        "pointer_relay": {
            "synchronous_batches": depth,
            "directed_transmissions": depth,
            "aggregate_payload_bits": relay_bits,
            "aggregate_payload_bytes_ascii": relay_bits,
        },
        "full_map_exchange": {
            "synchronous_batches": 1,
            "directed_transmissions": 2,
            "aggregate_payload_bits": full_map_bits,
            "aggregate_payload_bytes_ascii": full_map_bits,
        },
        "cost_model": "fixed-width binary-offset symbols serialized as unframed ASCII bits",
        "excluded_costs": ["transport framing", "prompt and schema", "tokenization", "inference", "setup"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": "tlu.pointer-chasing-oracle-frontier.v1",
        "model_calls": 0,
        "points": [oracle_frontier_point(n, k) for n in args.sizes for k in args.depths],
        "interpretation": "Exact analytic oracle costs; not a measured LLM or deployed transport frontier.",
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
