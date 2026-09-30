"""RFC 8259 string-safe labeled-field serialization challenger.

This is a model-free codec baseline, not a frozen prompt card or an LLM result.
It keeps the v0.3 key=value; framing but uses canonical JSON strings for keys
and values so delimiters, quotes, backslashes, control characters, and Unicode
do not require a newly invented escaping convention.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


def _require_unicode_scalars(value: str, *, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{label} contains an unpaired surrogate")


def _quote(value: str, *, label: str) -> str:
    _require_unicode_scalars(value, label=label)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def encode_fields(meaning: Mapping[str, str]) -> str:
    """Encode a string mapping in Unicode-codepoint key order."""
    if not isinstance(meaning, Mapping):
        raise TypeError("meaning must be a mapping")
    if not meaning:
        raise ValueError("meaning must contain at least one attribute")
    if any(not isinstance(key, str) for key in meaning):
        raise TypeError("attribute names must be strings")
    fields = []
    for key in sorted(meaning):
        value = meaning[key]
        if not isinstance(value, str):
            raise TypeError("attribute values must be strings")
        fields.append(f"{_quote(key, label='attribute name')}={_quote(value, label='attribute value')}")
    return ";".join(fields)


def _decode_json_string(message: str, position: int, *, label: str) -> tuple[str, int]:
    if position >= len(message) or message[position] != '"':
        raise ValueError(f"expected a quoted JSON {label}")
    try:
        value, end = json.JSONDecoder().raw_decode(message, position)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {label}") from exc
    if not isinstance(value, str):
        raise ValueError(f"JSON {label} must be a string")
    _require_unicode_scalars(value, label=label)
    return value, end


def decode_fields(message: str) -> dict[str, str]:
    """Decode canonical or noncanonical quoted fields; reject duplicates."""
    if not isinstance(message, str):
        raise TypeError("message must be text")
    _require_unicode_scalars(message, label="message")
    if not message:
        raise ValueError("message must contain at least one field")

    result: dict[str, str] = {}
    position = 0
    while position < len(message):
        key, position = _decode_json_string(message, position, label="attribute name")
        if position >= len(message) or message[position] != "=":
            raise ValueError("expected '=' after attribute name")
        value, position = _decode_json_string(message, position + 1, label="attribute value")
        if key in result:
            raise ValueError("duplicate attribute name")
        result[key] = value
        if position == len(message):
            break
        if message[position] != ";":
            raise ValueError("expected ';' between fields")
        position += 1
        if position == len(message):
            raise ValueError("message must not end with a separator")
    return result


def audit_fields(message: Any, target: Mapping[str, str]) -> dict[str, Any]:
    """Separate syntax, canonical encoding, key-set, and value fidelity."""
    result = {
        "syntactic_parse_valid": False,
        "exact_format_valid": False,
        "semantic_parse_valid": False,
        "canonical_label_fidelity": False,
        "failure_reason": None,
    }
    if not isinstance(message, str):
        result["failure_reason"] = "message is not text"
        return result
    try:
        decoded = decode_fields(message)
    except (TypeError, ValueError) as exc:
        result["failure_reason"] = str(exc)
        return result

    same_schema = isinstance(target, Mapping) and set(decoded) == set(target)
    canonical = False
    if same_schema:
        try:
            canonical = encode_fields(decoded) == message
        except (TypeError, ValueError):
            canonical = False
    result.update({
        "syntactic_parse_valid": True,
        "exact_format_valid": canonical,
        "semantic_parse_valid": same_schema,
        "canonical_label_fidelity": same_schema and dict(decoded) == dict(target),
    })
    if not same_schema:
        result["failure_reason"] = "parsed field labels do not match the target schema"
    elif not canonical:
        result["failure_reason"] = "message is valid but not in canonical form"
    elif not result["canonical_label_fidelity"]:
        result["failure_reason"] = "decoded tuple differs from target"
    return result


def compact_json_payload(meaning: Mapping[str, str]) -> str:
    """Return the matched compact JSON-object baseline with sorted keys."""
    for key, value in meaning.items():
        _require_unicode_scalars(key, label="attribute name")
        _require_unicode_scalars(value, label="attribute value")
    return json.dumps(meaning, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def compare_payload_bytes(meaning: Mapping[str, str]) -> dict[str, int]:
    """Count UTF-8 payload bytes without conflating them with model tokens."""
    labeled = encode_fields(meaning).encode("utf-8")
    compact_json = compact_json_payload(meaning).encode("utf-8")
    return {
        "quoted_labeled_fields_bytes": len(labeled),
        "compact_json_bytes": len(compact_json),
        "bytes_saved_vs_json": len(compact_json) - len(labeled),
    }
