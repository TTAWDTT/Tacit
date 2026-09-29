"""Portable frozen instruction cards for two-party text exchanges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping


PROTOCOL_CARD_SCHEMA = "tlu.shared_protocol_card.v1"
DIALOGUE_PROTOCOL_CARD_SCHEMA = "tlu.dialogue_protocol_card.v1"
_CARD_FIELDS = {"schema", "protocol_id", "sender_instruction", "receiver_instruction"}
_DIALOGUE_CARD_FIELDS = {"schema", "protocol_id", "agent_instructions"}
_MAX_PROTOCOL_ID_CHARS = 128
_MAX_INSTRUCTION_BYTES = 32_768
_MAX_AGENT_ROLES = 256
_MAX_AGENT_NAME_CHARS = 128


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_json(payload: str | bytes) -> Any:
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise TypeError("payload must be UTF-8 bytes or text")
        return json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("protocol card is not valid UTF-8 JSON") from exc
    except ValueError as exc:
        if str(exc).startswith("duplicate JSON field:"):
            raise
        raise ValueError("protocol card is not valid UTF-8 JSON") from exc


def _check_protocol_id(protocol_id: str) -> None:
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol_id must be a non-empty string")
    if len(protocol_id) > _MAX_PROTOCOL_ID_CHARS:
        raise ValueError("protocol_id is too long (maximum 128 characters)")


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
        _check_protocol_id(self.protocol_id)
        for field in ("sender_instruction", "receiver_instruction"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if len(value.encode("utf-8")) > _MAX_INSTRUCTION_BYTES:
                raise ValueError(f"{field} exceeds 32 KiB")

    @classmethod
    def from_json(cls, payload: str | bytes) -> "ProtocolCard":
        """Parse a schema-checked card; reject duplicate keys and extra fields."""
        data = _parse_json(payload)
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


@dataclass(frozen=True, init=False)
class DialogueProtocolCard:
    """Immutable role instructions for a scheduled multi-agent dialogue.

    The card fixes how named participants interpret their turns. The caller
    still fixes routes, turn order, stopping rules, and communication budget.
    """

    protocol_id: str
    _agent_instruction_items: tuple[tuple[str, str], ...]

    def __init__(self, protocol_id: str, agent_instructions: Mapping[str, str]) -> None:
        _check_protocol_id(protocol_id)
        if not isinstance(agent_instructions, Mapping):
            raise ValueError("agent_instructions must be a mapping")
        if not 2 <= len(agent_instructions) <= _MAX_AGENT_ROLES:
            raise ValueError("agent_instructions must define between 2 and 256 roles")
        items: list[tuple[str, str]] = []
        for name, instruction in agent_instructions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("agent role names must be non-empty strings")
            if len(name) > _MAX_AGENT_NAME_CHARS:
                raise ValueError("agent role names may not exceed 128 characters")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("each agent instruction must be a non-empty string")
            if len(instruction.encode("utf-8")) > _MAX_INSTRUCTION_BYTES:
                raise ValueError("agent instructions may not exceed 32 KiB per role")
            items.append((name, instruction))
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "_agent_instruction_items", tuple(sorted(items)))

    @property
    def agent_instructions(self) -> Mapping[str, str]:
        """Read-only instructions keyed by participating agent name."""
        return MappingProxyType(dict(self._agent_instruction_items))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "DialogueProtocolCard":
        """Parse a role-mapped card with strict schema and duplicate-key checks."""
        data = _parse_json(payload)
        if not isinstance(data, dict) or set(data) != _DIALOGUE_CARD_FIELDS:
            raise ValueError("dialogue protocol card fields are invalid")
        if data["schema"] != DIALOGUE_PROTOCOL_CARD_SCHEMA:
            raise ValueError("dialogue protocol card schema is invalid")
        if not isinstance(data["agent_instructions"], dict):
            raise ValueError("agent_instructions must be a JSON object")
        return cls(data["protocol_id"], data["agent_instructions"])

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned JSON object represented by this card."""
        return {
            "schema": DIALOGUE_PROTOCOL_CARD_SCHEMA,
            "protocol_id": self.protocol_id,
            "agent_instructions": dict(self._agent_instruction_items),
        }

    def to_json_bytes(self) -> bytes:
        """Serialize as deterministic compact UTF-8 JSON without a newline."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
