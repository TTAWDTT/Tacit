"""Versioned message formats for Private Match v0.3.

These are feasibility prompts, not an optimized natural-language baseline.
The protocol ID includes the frozen prompt/parser revision.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


PROMPT_REVISION = "pmt3-prompts-1"
PROTOCOL_IDS = ("concise_nl", "compact_kv", "strict_json", "fixed_binary")


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    code_id: str
    decoder_id: str
    sender_instruction: str
    receiver_instruction: str


def protocol_by_id(name: str, q: int) -> Protocol:
    if name not in PROTOCOL_IDS:
        raise ValueError(f"unknown protocol: {name}")
    if isinstance(q, bool) or not isinstance(q, int) or q < 2 or q & (q - 1):
        raise ValueError("q must be a power of two >= 2")
    width = q.bit_length() - 1
    sender_common = (
        "You are a coordinate source. Your private context contains exactly one coordinate. "
        "Communicate only that coordinate and its exact value; do not infer or invent the other coordinate. "
    )
    if name == "concise_nl":
        sender = sender_common + (
            'Write one short English sentence in this form: "The x coordinate is x0001." '
            'Replace x and the example value with the coordinate and value in your private context.'
        )
        receiver = (
            "Messages are short English statements from sender_x or sender_y. Extract the named coordinate "
            "and exact value from each message, match the pair to one candidate record, and return only its candidate_id."
        )
        code, decoder = "pmt3-concise-nl-v1", "pmt3-nl-coordinate-v1"
    elif name == "compact_kv":
        sender = sender_common + (
            'Return exactly one key-value string, such as x=x0001 or y=y0001. No spaces, punctuation, or explanation.'
        )
        receiver = (
            "Each message is exactly coordinate=value (for example x=x0001). Use the coordinate key and value, "
            "combine the two messages, match the candidate table, and return only candidate_id."
        )
        code, decoder = "pmt3-kv-v1", "pmt3-kv-coordinate-v1"
    elif name == "strict_json":
        sender = sender_common + (
            'Return exactly one JSON object with one key, either x or y, and its exact string value, '
            'for example {"x":"x0001"}. No markdown fences or additional keys.'
        )
        receiver = (
            "Each message is one JSON object with exactly one coordinate key (x or y) and one string value. "
            "Combine the two messages, match the candidate table, and return only candidate_id."
        )
        code, decoder = "pmt3-json-v1", "pmt3-json-coordinate-v1"
    else:
        sender = sender_common + (
            f"Encode the numeric suffix of your coordinate value as exactly {width} binary digits, preserving leading zeros. "
            "Return only those digits. The sender identity determines whether this is x or y."
        )
        receiver = (
            f"Each message contains exactly {width} binary digits. Interpret sender_x's digits as the x index and "
            "sender_y's digits as the y index; indices are zero-based and values are written xNNNN/yNNNN. "
            "Match the candidate table and return only candidate_id."
        )
        code, decoder = f"pmt3-fixed-{width}-bit-v1", f"pmt3-fixed-{width}-bit-decoder-v1"
    return Protocol(
        f"{name}:{PROMPT_REVISION}:q{q}",
        f"{PROMPT_REVISION}:{code}",
        f"{PROMPT_REVISION}:{decoder}",
        sender,
        receiver,
    )


def parse_coordinate_message(name: str, message: str, *, q: int, sender: str) -> tuple[bool | None, bool | None, str | None]:
    """Return (format-valid, semantic-fidelity-decidable, decoded-value).

    Natural-language fidelity is deliberately left undecidable without a
    separate blinded semantic judge; format validity is not used to score it.
    """
    coordinate = {"sender_x": "x", "sender_y": "y"}.get(sender)
    if coordinate is None or name not in PROTOCOL_IDS:
        raise ValueError("unknown protocol or sender")
    if name == "concise_nl":
        return None, None, None
    if name == "compact_kv":
        match = re.fullmatch(r"([xy])=(x|y)(\d{4})", message)
        if not match or match.group(1) != coordinate or match.group(2) != coordinate:
            return False, True, None
        index = int(match.group(3))
    elif name == "strict_json":
        try:
            value: Any = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return False, True, None
        if not isinstance(value, dict) or set(value) != {coordinate}:
            return False, True, None
        if message != json.dumps(value, ensure_ascii=True, separators=(",", ":")):
            return False, True, None
        raw = value[coordinate]
        if not isinstance(raw, str) or not re.fullmatch(rf"{coordinate}\d{{4}}", raw):
            return False, True, None
        index = int(raw[1:])
    else:
        width = q.bit_length() - 1
        if not re.fullmatch(rf"[01]{{{width}}}", message):
            return False, True, None
        index = int(message, 2)
    if not 0 <= index < q:
        return False, True, None
    return True, True, f"{coordinate}{index:04d}"
