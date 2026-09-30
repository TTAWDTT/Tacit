"""Model-facing role cards for the exhaustive finite private-sum codebooks."""
from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from typing import Any, Mapping

from tacit import DialogueProtocolCard, exchange_dialogue

from examples.multiparty_private_sum import MAX_REQUESTS_PER_EPISODE, SUM_TASK
from research.multiparty_sum_lossy_frontier import exact_frontier


RECEIVER = "R"
SCHEMA = "tlu.multiparty-sum-codebook-card.v0.1.0"
PRIOR = "independent_uniform_integer_0_to_3"
SUM_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _width(partition: list[list[int]]) -> int:
    return (len(partition) - 1).bit_length()


def _word(message_id: int, width: int) -> str:
    return format(message_id, f"0{width}b") if width else ""


def _protocol_id(agent_count: int, partitions: list[list[list[int]]]) -> str:
    source = json.dumps(
        {"agent_count": agent_count, "partitions": partitions},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return "sum-code-" + hashlib.sha256(source).hexdigest()[:12]


def _message_id(value: int, partition: list[list[int]]) -> int:
    for index, block in enumerate(partition):
        if value in block:
            return index
    raise ValueError("private value is not represented by the sender's partition")


def _receiver_instruction(
    agent_count: int,
    partitions: list[list[list[int]]],
    decoder_rows: list[dict[str, Any]],
) -> str:
    widths = [_width(partition) for partition in partitions]
    active = [i + 1 for i, width in enumerate(widths) if width > 0]
    table = "\n".join(
        f"{','.join(map(str, row['message_ids']))} -> {row['decoded_sum']}"
        for row in decoder_rows
    )
    inactive = [i + 1 for i, width in enumerate(widths) if width == 0]
    codeword_tables = []
    for index, (partition, width) in enumerate(zip(partitions, widths), start=1):
        if width == 0:
            codeword_tables.append(f"S{index}: omitted; implicit message ID 0")
        else:
            labels = "; ".join(
                f"{_word(message_id, width)} = ID {message_id}"
                for message_id in range(len(partition))
            )
            codeword_tables.append(f"S{index}: {labels}")
    return (
        "You are the referee for a simultaneous private-input sum task. The prior is independent uniform integers "
        "in {0,1,2,3}. Public sender_count is " + str(agent_count) + ".\n"
        "Only active senders " + (", ".join(f"S{i}" for i in active) if active else "none")
        + " send one fixed-width binary message each. Decode codewords using this table:\n"
        + "\n".join(codeword_tables) + "\n"
        + (f"Senders {', '.join(f'S{i}' for i in inactive)} are omitted because their encoder has one constant "
           "message ID 0. This does NOT mean their private input is zero; their unknown values remain in the prior.\n"
           if inactive else "")
        + "Use this complete MAP table. A row maps the ordered tuple of message IDs (S1 through S"
        + str(agent_count) + ") to the optimal exact-sum guess. For omitted senders, insert ID 0.\n"
        + table + "\nReturn only the selected non-negative integer sum."
    )


def build_protocol_card(agent_count: int, payload_budget_bits: int) -> dict[str, Any]:
    """Build a frozen LLM role card and all setup-size measurements.

    Singleton encoders are omitted from the call/message schedule. Their private
    values remain unknown; the receiver inserts their constant message ID 0.
    The B=0 point has no active sender and is represented by the no-message arm.
    """
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or not 2 <= agent_count <= 4:
        raise ValueError("agent_count must be between 2 and 4 for the exhaustive codebook family")
    if isinstance(payload_budget_bits, bool) or not isinstance(payload_budget_bits, int) or not 0 <= payload_budget_bits <= 2 * agent_count:
        raise ValueError("payload_budget_bits is outside this agent-count frontier")
    frontier = exact_frontier(agent_count)
    row = frontier[payload_budget_bits]
    partitions = row["optimal_sender_partitions"]
    widths = [_width(partition) for partition in partitions]
    active_senders = [f"S{index + 1}" for index, width in enumerate(widths) if width > 0]
    if not active_senders:
        raise ValueError("the zero-bit point has no codebook senders; use the no_message condition")
    instructions: dict[str, str] = {}
    for index, (partition, width) in enumerate(zip(partitions, widths), start=1):
        if width == 0:
            continue
        mappings = []
        for message_id, block in enumerate(partition):
            values = ",".join(map(str, block))
            mappings.append(f"{values} -> {_word(message_id, width)}")
        instructions[f"S{index}"] = (
            f"You are sender S{index}. The task is to help a referee find the exact sum of all {agent_count} "
            f"private integers, each drawn independently and uniformly from {{0,1,2,3}}. Your only private value is "
            "the integer in your private context. Apply this fixed encoder table exactly:\n"
            + "\n".join(mappings)
            + f"\nTransmit only the mapped {width}-bit codeword. Do not send the private integer, explanation, or any other text."
        )
    instructions[RECEIVER] = _receiver_instruction(
        agent_count, partitions, row["optimal_receiver_decoder"],
    )
    protocol_id = _protocol_id(agent_count, partitions)
    card = DialogueProtocolCard(protocol_id, instructions)
    card_bytes = card.to_json_bytes()
    role_instruction_bytes = {
        role: len(text.encode("utf-8")) for role, text in card.agent_instructions.items()
    }
    return {
        "schema_version": SCHEMA,
        "protocol_id": protocol_id,
        "agent_count": agent_count,
        "payload_budget_bits_at_most": payload_budget_bits,
        "actual_fixed_width_payload_bits": row["protocol_minimum_bits"],
        "optimal_exact_sum_success": row["optimal_exact_sum_success"],
        "optimal_success_numerator": row["optimal_success_numerator"],
        "input_vector_count": row["input_vector_count"],
        "sender_partitions": partitions,
        "sender_widths_bits": widths,
        "active_senders": active_senders,
        "omitted_sender_indices": [i + 1 for i, width in enumerate(widths) if width == 0],
        "decoder_entries": len(row["optimal_receiver_decoder"]),
        "agent_instructions": dict(card.agent_instructions),
        "dialogue_protocol_card": card.to_dict(),
        "card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        "card_json_utf8_bytes": len(card_bytes),
        "role_instruction_utf8_bytes": role_instruction_bytes,
        "total_role_instruction_utf8_bytes": sum(role_instruction_bytes.values()),
        "system_instruction_utf8_bytes_per_episode": sum(role_instruction_bytes.values()),
        "setup_amortization_utf8_bytes_per_episode": {
            str(horizon): len(card_bytes) / horizon for horizon in (1, 10, 100, 1000)
        },
        "setup_token_cost": "model/tokenizer-specific; measure provider input tokens on an actual run",
    }


def _parse_sum(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not SUM_RE.fullmatch(stripped):
        return None
    return int(stripped)


def _oracle_sum_for_observed_messages(
    partitions: list[list[list[int]]], turns: tuple[Any, ...],
) -> tuple[int, tuple[int, ...], list[dict[str, Any]]]:
    decoded_ids: list[int] = []
    diagnostics = []
    delivered = {turn.speaker: turn.completion.text for turn in turns if turn.transmission is not None}
    for index, partition in enumerate(partitions, start=1):
        sender = f"S{index}"
        width = _width(partition)
        if width == 0:
            decoded_ids.append(0)
            diagnostics.append({"sender": sender, "status": "implicit_singleton_id", "message_id": 0})
            continue
        message = delivered.get(sender)
        if message is None or len(message) != width or any(bit not in "01" for bit in message):
            decoded_ids.append(-1)
            diagnostics.append({"sender": sender, "status": "missing_or_malformed", "message_id": None})
            continue
        message_id = int(message, 2)
        if message_id >= len(partition):
            decoded_ids.append(-1)
            diagnostics.append({"sender": sender, "status": "unused_codeword", "message_id": None})
            continue
        decoded_ids.append(message_id)
        diagnostics.append({"sender": sender, "status": "valid", "message_id": message_id})
    coefficients = [1]
    for message_id, partition in zip(decoded_ids, partitions):
        values = set(range(4)) if message_id < 0 else partition[message_id]
        updated = [0] * (len(coefficients) + max(values))
        for subtotal, count in enumerate(coefficients):
            for value in values:
                updated[subtotal + value] += count
        coefficients = updated
    mode = coefficients.index(max(coefficients))
    return mode, tuple(decoded_ids), diagnostics


def run_codebook_episode(
    values: list[int],
    *,
    card_spec: Mapping[str, Any],
    sender_clients: Mapping[str, Any],
    receiver_client: Any,
    wire_budget_bytes: int = 4096,
    request_cap: int = MAX_REQUESTS_PER_EPISODE,
) -> dict[str, Any]:
    """Use an optimal partition card in the real Tacit dialogue runtime."""
    if not isinstance(values, list) or not 2 <= len(values) <= 4:
        raise ValueError("values must contain between two and four private sender integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in values):
        raise ValueError("each private value must be an integer in {0,1,2,3}")
    if card_spec.get("agent_count") != len(values):
        raise ValueError("codebook card agent_count does not match the task")
    canonical_card = build_protocol_card(len(values), card_spec.get("payload_budget_bits_at_most"))
    if dict(card_spec) != canonical_card:
        raise ValueError("codebook card differs from the canonical exhaustive frontier artifact")
    instructions = card_spec.get("agent_instructions")
    active = list(card_spec.get("active_senders", []))
    if not isinstance(instructions, Mapping) or set(instructions) != {*active, RECEIVER}:
        raise ValueError("codebook role instructions do not match active sender schedule")
    if set(sender_clients) != set(active):
        raise ValueError("sender_clients must contain exactly one client for every active sender")
    if isinstance(request_cap, bool) or not isinstance(request_cap, int) or not 1 <= request_cap <= MAX_REQUESTS_PER_EPISODE:
        raise ValueError(f"request_cap must be between 1 and {MAX_REQUESTS_PER_EPISODE}")
    if isinstance(wire_budget_bytes, bool) or not isinstance(wire_budget_bytes, int) or not 0 <= wire_budget_bytes <= 4096:
        raise ValueError("wire_budget_bytes must be between zero and 4096")
    planned_calls = len(active) + 1
    if planned_calls > request_cap:
        raise ValueError(f"codebook episode requires {planned_calls} calls, exceeding request cap {request_cap}")

    card = DialogueProtocolCard(card_spec["protocol_id"], instructions)
    agents = {**dict(sender_clients), RECEIVER: receiver_client}
    contexts = {
        **{
            f"S{index + 1}": f"Public metadata: sender_count={len(values)}. Your private integer is {value}."
            for index, value in enumerate(values) if f"S{index + 1}" in active
        },
        RECEIVER: f"Public metadata: sender_count={len(values)}. Assumed input prior: {PRIOR}.",
    }
    schedule = tuple((sender, RECEIVER) for sender in active)
    result = exchange_dialogue(
        agents,
        protocol=card,
        private_contexts=contexts,
        schedule=schedule,
        task=SUM_TASK,
        max_turns=len(active),
        wire_budget_bytes=wire_budget_bytes,
        final_answer_agent=RECEIVER,
        final_answer_instruction="Return only the exact non-negative integer sum as base-10 digits.",
    )
    expected = sum(values)
    oracle_prediction, decoded_ids, sender_diagnostics = _oracle_sum_for_observed_messages(
        card_spec["sender_partitions"], result.turns,
    )
    raw_output = "" if result.final_submission is None else result.final_submission.text
    prediction = _parse_sum(raw_output)
    sender_rows = []
    for diagnostic in sender_diagnostics:
        index = int(diagnostic["sender"][1:]) - 1
        partition = card_spec["sender_partitions"][index]
        true_id = _message_id(values[index], partition)
        received_id = diagnostic["message_id"]
        turn = next((item for item in result.turns if item.speaker == diagnostic["sender"]), None)
        sender_rows.append({
            **diagnostic,
            "expected_codeword": _word(true_id, _width(partition)) if _width(partition) else None,
            "sender_value_represented_by_cell": (
                None if received_id is None else values[index] in partition[received_id]
            ),
            "generated_message": None if turn is None else turn.completion.text,
            "delivered": bool(turn is not None and turn.transmission is not None),
            "application_bytes": None if turn is None or turn.transmission is None else turn.transmission.total_application_bytes,
        })
    return {
        "protocol_id": result.protocol_id,
        "task": "uniform_four_value_private_sum",
        "agent_count": len(values),
        "payload_budget_bits_at_most": card_spec["payload_budget_bits_at_most"],
        "actual_fixed_width_payload_bits": card_spec["actual_fixed_width_payload_bits"],
        "optimal_exact_sum_success_reference": card_spec["optimal_exact_sum_success"],
        "codebook_card_sha256": card_spec["card_sha256"],
        "codebook_card_json_utf8_bytes": card_spec["card_json_utf8_bytes"],
        "system_instruction_utf8_bytes_per_episode": card_spec["system_instruction_utf8_bytes_per_episode"],
        "sender_partitions": card_spec["sender_partitions"],
        "active_senders": active,
        "omitted_sender_indices": card_spec["omitted_sender_indices"],
        "expected_sum": expected,
        "prediction": prediction,
        "exact_success": prediction == expected,
        "oracle_prediction_for_observed_transcript": oracle_prediction,
        "receiver_matches_codebook_oracle": prediction == oracle_prediction,
        "all_delivered_sender_messages_match_partition": (
            None if not any(row["delivered"] for row in sender_rows) else all(
                row["sender_value_represented_by_cell"] is True
                for row in sender_rows if row["delivered"]
            )
        ),
        "sender_messages": sender_rows,
        "model_calls": result.model_calls,
        "planned_model_calls": planned_calls,
        "wire_bytes": result.wire_bytes,
        "wire_budget_bytes": result.wire_budget_bytes,
        "stop_reason": result.stop_reason,
        "message_transmissions": result.transmission_records(),
        "model_call_records": result.model_call_records(),
        "raw_answer": raw_output,
    }
