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


def parity_assisted_protocol(episode: dict) -> dict:
    """Exact sequential (k-1)-turn protocol for k >= 3.

    Each agent includes its n-bit function-parity table in its first pointer
    message. The agents then exchange the first k-1 pointers. Both know p_(k-1)
    and the parity table of the final map owner, so both derive parity(p_k)
    without a final pointer message.
    """
    validate_episode(episode)
    size, depth = episode["size"], episode["depth"]
    if depth < 3:
        raise ValueError("parity-assisted (k-1)-turn control requires depth k >= 3")
    width = (size - 1).bit_length()
    functions = {
        "agent_a": episode["agent_a_view"]["function_values_1_based"],
        "agent_b": episode["agent_b_view"]["function_values_1_based"],
    }
    parity_tables = {role: "".join(str(value % 2) for value in values) for role, values in functions.items()}
    known_at_a = {"agent_a": parity_tables["agent_a"]}
    known_at_b = {"agent_b": parity_tables["agent_b"]}
    pointer = 1
    messages = []
    for step in range(1, depth):
        sender = "agent_a" if step % 2 else "agent_b"
        function = functions[sender]
        pointer = function[pointer - 1]
        pointer_payload = encode_pointer(pointer, size)
        pointer = decode_pointer(pointer_payload, size)
        parity_payload = parity_tables[sender] if step <= 2 else ""
        if parity_payload:
            # The sender knows its own table; the receiver learns it in this message.
            known_at_a[sender] = parity_payload
            known_at_b[sender] = parity_payload
        messages.append({
            "step": step,
            "sender": sender,
            "parity_table_bits": parity_payload,
            "pointer_bits": pointer_payload,
            "payload_bits": parity_payload + pointer_payload,
        })

    final_owner = "agent_a" if depth % 2 else "agent_b"
    if final_owner not in known_at_a or final_owner not in known_at_b:
        raise AssertionError("the final map owner's parity table was not shared")
    answer_a = int(known_at_a[final_owner][pointer - 1])
    answer_b = int(known_at_b[final_owner][pointer - 1])
    if answer_a != answer_b:
        raise AssertionError("agents computed different answers from the shared final parity table")
    if answer_a != episode["gold_bit"]:
        raise AssertionError("parity-assisted protocol disagrees with the task answer")
    payload_bits = sum(len(message["payload_bits"]) for message in messages)
    expected_bits = 2 * size + (depth - 1) * width
    if payload_bits != expected_bits:
        raise AssertionError("parity-assisted transcript does not match its exact bit formula")
    return {
        "policy_id": "parity-assisted-skip-final-pointer-v1",
        "answer_bit": answer_a,
        "agent_answers": {"agent_a": answer_a, "agent_b": answer_b},
        "both_agents_independently_compute_answer": True,
        "sequential_speaker_turns": depth - 1,
        "directed_transmissions": depth - 1,
        "bits_per_function_value": width,
        "parity_table_bits_per_agent": size,
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
    parity_assisted = None if depth < 3 else 2 * size + (depth - 1) * width
    return {
        "size": size,
        "depth": depth,
        "zero_error": True,
        "both_agents_independently_compute_answer": True,
        "pointer_relay": {
            "sequential_speaker_turns": depth,
            "directed_transmissions": depth,
            "aggregate_payload_bits": relay_bits,
            "aggregate_payload_bytes_ascii": relay_bits,
        },
        "full_map_exchange": {
            "simultaneous_batches": 1,
            "sequential_speaker_turns": 0,
            "directed_transmissions": 2,
            "aggregate_payload_bits": full_map_bits,
            "aggregate_payload_bytes_ascii": full_map_bits,
        },
        "parity_assisted_skip_final_pointer": {
            "eligible": depth >= 3,
            "sequential_speaker_turns": depth - 1 if depth >= 3 else None,
            "directed_transmissions": depth - 1 if depth >= 3 else None,
            "aggregate_payload_bits": parity_assisted,
            "aggregate_payload_bytes_ascii": parity_assisted,
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
        "schema_version": "tlu.pointer-chasing-oracle-frontier.v2",
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
