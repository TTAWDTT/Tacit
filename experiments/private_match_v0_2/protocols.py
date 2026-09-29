"""Pre-registered message-format conditions for Private Match v0.2."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    code_id: str
    decoder_id: str
    sender_instruction: str
    receiver_instruction: str


_OUTPUT = "Return only the answer requested by the receiver task. No explanation."


def protocol_by_id(name: str) -> Protocol:
    if name == "no_message":
        return Protocol(
            name, "none", "no-message-prior-v1",
            "No message is sent in this condition.",
            "You receive no message from the sender. Use only the candidate table and return one candidate_id. The target is independent of the table, so do not invent evidence.",
        )
    if name == "concise_nl":
        return Protocol(
            name, "prompt-v1", "candidate-match-v1",
            "Send the target record to another agent as a short natural-language message. Include every feature name and value exactly. Do not include any other information.",
            f"The message describes a record. Match every feature and exact value against the candidate table, then return the matching candidate_id. {_OUTPUT}",
        )
    if name == "json":
        return Protocol(
            name, "prompt-v1", "json-record-match-v1",
            'Serialize the target record as one compact JSON object. Preserve every key and value exactly. Return only the JSON object.',
            f"Interpret the message as a JSON object containing feature/value pairs. Find the exact record match in the table and return its candidate_id. {_OUTPUT}",
        )
    if name == "compact_kv":
        return Protocol(
            name, "fixed-kv-v1", "ordered-kv-exact-match-v1",
            "Encode every feature as name=value, in the stated feature order, with pairs separated by semicolons and no spaces. Use the exact feature names and values. Return only the pairs, for example f0=v0002;f1=v0010.",
            f"Interpret the message as semicolon-separated name=value pairs. Match the exact values against the candidate table and return the matching candidate_id. {_OUTPUT}",
        )
    if name == "tuple":
        return Protocol(
            name, "prompt-v1", "ordered-tuple-match-v1",
            "Send only a comma-separated tuple of values, in the feature order stated in the task. Do not include labels or spaces.",
            f"Interpret the message as comma-separated values in the feature order stated in the task. Match the exact tuple and return candidate_id. {_OUTPUT}",
        )
    if name == "hex_nibbles":
        return Protocol(
            name, "fixed-codebook-v1", "hex-nibble-decode-v1",
            "For each feature in the stated feature order, convert its value vNNNN to the integer NNNN, encode that integer as exactly one lowercase hexadecimal digit, then concatenate the digits. Return only the resulting code. The vocabulary has values v0000 through v0015.",
            f"Decode each lowercase hexadecimal digit as an integer 0 through 15, map it to the corresponding value v0000 through v0015, and assign digits to features in the stated order. Match the resulting record and return candidate_id. {_OUTPUT}",
        )
    if name == "autoform":
        return Protocol(
            name, "prompt-v1", "sender-selected-form-match-v1",
            "Choose any compact representation you expect another capable language model to decode reliably. Include all feature names and exact values, and send only the message.",
            f"The sender chose a representation for a small record. Interpret its content, match all feature/value pairs to the exact candidate row, and return candidate_id. {_OUTPUT}",
        )
    raise ValueError(f"unknown protocol: {name}")


PROTOCOL_IDS = ("concise_nl", "compact_kv", "json", "tuple", "hex_nibbles", "autoform")
