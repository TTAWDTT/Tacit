"""Portable frozen instruction cards for two-party text exchanges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


PROTOCOL_CARD_SCHEMA = "tlu.shared_protocol_card.v1"
_CARD_FIELDS = {"schema", "protocol_id", "sender_instruction", "receiver_instruction"}
_MAX_PROTOCOL_ID_CHARS = 128
_MAX_INSTRUCTION_BYTES = 32_768


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ProtocolCard:
    """Immutable, portable sender/receiver instructions for ``exchange_once``.

    A card fixes prompt instructions, not model behavior or semantics. The
    receiver and sender still interpret instructions using their own models.
    """

    protocol_id: str
    sender_instruction: str
    receiver_instruction: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError("protocol_id must be a non-empty string")
        if len(self.protocol_id) > _MAX_PROTOCOL_ID_CHARS:
            raise ValueError("protocol_id is too long (maximum 128 characters)")
        for field in ("sender_instruction", "receiver_instruction"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if len(value.encode("utf-8")) > _MAX_INSTRUCTION_BYTES:
                raise ValueError(f"{field} exceeds 32 KiB")

    @classmethod
    def from_json(cls, payload: str | bytes) -> "ProtocolCard":
        """Parse a schema-checked card; reject duplicate keys and extra fields."""
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            if not isinstance(payload, str):
                raise TypeError("payload must be UTF-8 bytes or text")
            data = json.loads(payload, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("protocol card is not valid UTF-8 JSON") from exc
        except ValueError as exc:
            if str(exc).startswith("duplicate JSON field:"):
                raise
            raise ValueError("protocol card is not valid UTF-8 JSON") from exc
        if not isinstance(data, dict) or set(data) != _CARD_FIELDS:
            raise ValueError("protocol card fields are invalid")
        if data["schema"] != PROTOCOL_CARD_SCHEMA:
            raise ValueError("protocol card schema is invalid")
        return cls(
            protocol_id=data["protocol_id"],
            sender_instruction=data["sender_instruction"],
            receiver_instruction=data["receiver_instruction"],
        )

    @classmethod
    def read(cls, path: str) -> tuple["ProtocolCard", bytes]:
        """Read a card and return it with the exact source bytes for provenance hashing."""
        from pathlib import Path

        try:
            source = Path(path).read_bytes()
        except OSError as exc:
            raise ValueError("protocol card could not be read") from exc
        return cls.from_json(source), source

    def to_dict(self) -> dict[str, str]:
        """Return the complete versioned JSON object represented by this card."""
        return {
            "schema": PROTOCOL_CARD_SCHEMA,
            "protocol_id": self.protocol_id,
            "sender_instruction": self.sender_instruction,
            "receiver_instruction": self.receiver_instruction,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize deterministically as compact UTF-8 JSON (without a newline)."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
