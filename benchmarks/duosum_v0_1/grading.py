"""Deterministic message and answer scoring helpers for DuoSum.

This module deliberately does not replace an upstream benchmark score. It
provides separate, auditable syntax/adherence and semantic-answer diagnostics.
"""

from __future__ import annotations

import json
import re
from typing import Any


EQUATION_RE = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")


def parse_answer(answer: Any) -> tuple[int | None, str]:
    """Return a verified numeric answer and its serialization class."""
    if isinstance(answer, int) and not isinstance(answer, bool):
        return answer, "integer"
    if not isinstance(answer, str):
        return None, "unparseable"
    text = answer.strip()
    if text.isdecimal():
        return int(text), "integer_string"
    match = EQUATION_RE.fullmatch(text)
    if match is None:
        return None, "unparseable"
    left, right, stated = map(int, match.groups())
    if left + right != stated:
        return None, "invalid_equation"
    return stated, "verified_equation"


def answer_is_correct(answer: Any, expected: int) -> bool:
    """Check a raw answer without rewriting the benchmark's strict score."""
    parsed, _kind = parse_answer(answer)
    return parsed == expected


def decode_message(content: str, representation: str) -> int | None:
    """Decode the supported fixed representations; return None on mismatch."""
    text = content.strip()
    if representation == "concise_nl":
        match = re.fullmatch(r"My private value is (\d+)", text)
        return int(match.group(1)) if match else None
    if representation == "compact_kv":
        match = re.fullmatch(r"v=(\d+)", text)
        return int(match.group(1)) if match else None
    if representation == "json_schema":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(value, dict)
            and set(value) == {"v"}
            and isinstance(value["v"], int)
            and not isinstance(value["v"], bool)
        ):
            return value["v"]
        return None
    if representation == "binary":
        if re.fullmatch(r"[01]+", text) is None:
            return None
        value = int(text, 2)
        return value if text == format(value, "b") else None
    return None


def message_matches(content: str, representation: str, expected_value: int) -> bool:
    """Return whether a message both decodes and conveys its sender's value."""
    return decode_message(content, representation) == expected_value
