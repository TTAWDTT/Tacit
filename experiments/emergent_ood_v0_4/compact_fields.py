"""Scoring helpers for the frozen ontology-general compact fields card."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_ID = "compact-labeled-fields-generic-v3"
RESERVED_DELIMITERS = (";", "=")


def encode_fields(meaning: Mapping[str, str]) -> str:
    """Serialize a schema mapping in canonical key order without a codebook."""
    if not isinstance(meaning, Mapping) or len(meaning) < 2:
        raise ValueError("meaning must be a mapping with at least two attributes")
    if any(not isinstance(key, str) or not key.strip() for key in meaning):
        raise ValueError("attribute names must be non-empty strings")
    if any(any(delimiter in key for delimiter in RESERVED_DELIMITERS) for key in meaning):
        raise ValueError("attribute names cannot contain grammar delimiters")

    fields = []
    for attribute in sorted(meaning):
        value = meaning[attribute]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("attribute values must be non-empty strings")
        if any(delimiter in value for delimiter in RESERVED_DELIMITERS):
            raise ValueError("attribute values cannot contain grammar delimiters")
        if attribute != attribute.strip() or value != value.strip():
            raise ValueError("attribute names and values cannot have surrounding whitespace")
        fields.append(f"{attribute}={value}")
    return ";".join(fields)


def audit_fields(message: Any, target: Mapping[str, str]) -> dict[str, Any]:
    """Score syntax, canonical ordering, schema parse, and exact tuple fidelity."""
    empty = {
        "syntactic_parse_valid": False,
        "exact_format_valid": False,
        "semantic_parse_valid": False,
        "canonical_label_fidelity": False,
    }
    if not isinstance(message, str):
        return {**empty, "failure_reason": "message is not text"}
    if (
        not isinstance(target, Mapping)
        or len(target) < 2
        or any(
            not isinstance(key, str) or not key.strip() or key != key.strip()
            for key in target
        )
    ):
        raise ValueError("target tuple has an invalid attribute schema")
    if any(any(delimiter in key for delimiter in RESERVED_DELIMITERS) for key in target):
        raise ValueError("target schema contains unsupported grammar delimiters")
    for value in target.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("target tuple has an empty or non-text value")
        if any(delimiter in value for delimiter in RESERVED_DELIMITERS):
            raise ValueError("target tuple contains an unsupported grammar delimiter")
        if value != value.strip():
            raise ValueError("target values cannot have surrounding whitespace")

    pieces = message.split(";")
    decoded: dict[str, str] = {}
    for piece in pieces:
        if piece.count("=") != 1:
            return {**empty, "failure_reason": "field must contain exactly one equals sign"}
        key, value = piece.split("=", 1)
        if not key or not value or key != key.strip() or value != value.strip():
            return {**empty, "failure_reason": "field has an empty value or surrounding whitespace"}
        if key in decoded:
            return {**empty, "failure_reason": "duplicate attribute label"}
        decoded[key] = value

    expected_keys = set(target)
    parse_valid = len(pieces) == len(target) and set(decoded) == expected_keys
    format_valid = parse_valid and list(decoded) == sorted(expected_keys)
    fidelity = parse_valid and decoded == dict(target)
    if not parse_valid:
        reason = "parsed field labels do not match the target schema"
    elif not format_valid:
        reason = "field labels are not in canonical order"
    elif not fidelity:
        reason = "decoded tuple differs from target"
    else:
        reason = None
    return {
        "syntactic_parse_valid": True,
        "exact_format_valid": format_valid,
        "semantic_parse_valid": parse_valid,
        "canonical_label_fidelity": fidelity,
        "failure_reason": reason,
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
        if not isinstance(target, Mapping):
            raise ValueError(f"result row {index} has no evaluator target tuple")
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
