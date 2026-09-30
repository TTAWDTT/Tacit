"""Scoring helpers for the frozen compact labeled-fields card."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_ID = "compact-labeled-fields-default-v1"
ATTRIBUTES = ("shape", "color", "quantity", "texture")
VALUES_BY_ATTRIBUTE = {
    "shape": ("circle", "square", "triangle", "hexagon"),
    "color": ("red", "blue", "green", "yellow"),
    "quantity": ("one", "two", "three", "four"),
    "texture": ("smooth", "rough", "striped", "dotted"),
}


def encode_fields(meaning: Mapping[str, str]) -> str:
    """Return the canonical default-ontology payload; reject unescaped values."""
    if not isinstance(meaning, Mapping) or set(meaning) != set(ATTRIBUTES):
        raise ValueError("meaning must contain exactly the default ontology attributes")
    fields = []
    for attribute in ATTRIBUTES:
        value = meaning[attribute]
        if not isinstance(value, str) or value not in VALUES_BY_ATTRIBUTE[attribute]:
            raise ValueError(f"unknown value for {attribute}")
        fields.append(f"{attribute}={value}")
    return ";".join(fields)


def audit_fields(message: Any, target: Mapping[str, str]) -> dict[str, Any]:
    """Separate strict syntax, ontology decoding, and sender-target fidelity."""
    if not isinstance(message, str):
        return {
            "exact_format_valid": False,
            "semantic_parse_valid": False,
            "canonical_label_fidelity": False,
            "failure_reason": "message is not text",
        }

    pieces = message.split(";")
    if len(pieces) != len(ATTRIBUTES):
        return {
            "exact_format_valid": False,
            "semantic_parse_valid": False,
            "canonical_label_fidelity": False,
            "failure_reason": "wrong field count",
        }

    decoded: dict[str, str] = {}
    for attribute, piece in zip(ATTRIBUTES, pieces):
        if piece.count("=") != 1:
            return {
                "exact_format_valid": False,
                "semantic_parse_valid": False,
                "canonical_label_fidelity": False,
                "failure_reason": "field must contain exactly one equals sign",
            }
        key, value = piece.split("=", 1)
        if key != attribute or not value or any(char.isspace() for char in piece):
            return {
                "exact_format_valid": False,
                "semantic_parse_valid": False,
                "canonical_label_fidelity": False,
                "failure_reason": "field label, order, or spacing is noncanonical",
            }
        decoded[attribute] = value

    semantic_valid = all(
        decoded[attribute] in VALUES_BY_ATTRIBUTE[attribute]
        for attribute in ATTRIBUTES
    )
    target_valid = (
        isinstance(target, Mapping)
        and set(target) == set(ATTRIBUTES)
        and all(target.get(attribute) in VALUES_BY_ATTRIBUTE[attribute] for attribute in ATTRIBUTES)
    )
    fidelity = semantic_valid and target_valid and decoded == dict(target)
    return {
        "exact_format_valid": True,
        "semantic_parse_valid": semantic_valid,
        "canonical_label_fidelity": fidelity,
        "failure_reason": None if fidelity else (
            "value outside ontology" if not semantic_valid else "decoded tuple differs from target"
        ),
    }


def audit_result_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Audit raw v0.4 shared-card result rows without modifying their source."""
    audited = []
    seen_episode_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"result row {index} is not an object")
        if row.get("condition") != "shared_protocol_card" or row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"result row {index} is not from the frozen compact-fields card")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in seen_episode_ids:
            raise ValueError(f"result row {index} has a missing or duplicate episode_id")
        seen_episode_ids.add(episode_id)
        trace = row.get("trace")
        if not isinstance(trace, Mapping) or "message" not in trace:
            raise ValueError(f"result row {index} has no raw sender message")
        target = trace.get("target_tuple_for_evaluator")
        if not isinstance(target, Mapping) or set(target) != set(ATTRIBUTES):
            raise ValueError(f"result row {index} has no valid evaluator target tuple")
        message = trace["message"]
        result = audit_fields(message, target)
        message_bytes = len(message.encode("utf-8")) if isinstance(message, str) else None
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping) or not isinstance(outcome.get("exact_selection"), bool):
            raise ValueError(f"result row {index} has no boolean exact_selection outcome")
        audited.append({
            "episode_id": episode_id,
            "stage": row.get("stage"),
            "message_bytes": message_bytes,
            **result,
            "exact_selection": outcome["exact_selection"],
        })
    if not audited:
        raise ValueError("input contains no result rows")
    return audited
