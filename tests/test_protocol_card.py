from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from tacit import (
    ChatCompletion,
    DialogueProtocolCard,
    ProtocolCard,
    exchange_dialogue,
    exchange_once,
)


class _Model:
    def __init__(self, text: str, name: str) -> None:
        self.text = text
        self.name = name
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        return ChatCompletion(self.text, self.name)


class ProtocolCardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.card = ProtocolCard(
            "portable-card-v1", "Transmit relevant facts.", "Use received facts."
        )

    def test_compact_utf8_round_trip_and_exchange_compatibility(self) -> None:
        card = ProtocolCard("portable-v1", "Send café facts.", "Use café facts.")
        encoded = card.to_json_bytes()
        self.assertIn("café".encode(), encoded)
        self.assertEqual(ProtocolCard.from_json(encoded), card)

        sender = _Model("evidence", "sender")
        receiver = _Model("answer", "receiver")
        result = exchange_once(
            sender, receiver, protocol=card, sender_context="private A",
            receiver_context="private B", receiver_task="combine",
        )
        self.assertEqual(result.protocol_id, "portable-v1")
        self.assertEqual(sender.calls[0][0]["content"], card.sender_instruction)
        self.assertEqual(receiver.calls[0][0]["content"], card.receiver_instruction)
        self.assertEqual(result.message, "evidence")

    def test_card_is_immutable_and_serialization_is_deterministic(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.card.protocol_id = "mutated"
        self.assertEqual(self.card.to_json_bytes(), self.card.to_json_bytes())
        self.assertEqual(json.loads(self.card.to_json_bytes()), self.card.to_dict())

    def test_shipped_example_card_uses_the_public_schema(self) -> None:
        card_path = Path(__file__).resolve().parents[1] / "examples" / "protocol_card.json"
        card = ProtocolCard.from_json(card_path.read_bytes())
        self.assertEqual(card.protocol_id, "plain-text-evidence-review-example-v1")

    def test_dialogue_card_is_immutable_and_works_with_three_agent_runtime(self) -> None:
        source = {name: f"Use private context and routed messages ({name})." for name in "ABC"}
        card = DialogueProtocolCard("three-role-card-v1", source)
        source["A"] = "mutated after construction"
        self.assertNotEqual(card.agent_instructions["A"], source["A"])
        with self.assertRaises(TypeError):
            card.agent_instructions["A"] = "mutated"
        self.assertEqual(DialogueProtocolCard.from_json(card.to_json_bytes()), card)

        agents = {name: _Model(f"message-{name}", f"model-{name}") for name in "ABC"}
        result = exchange_dialogue(
            agents,
            protocol=card,
            private_contexts={name: f"private-{name}" for name in "ABC"},
            schedule=(("A", "B"), ("B", "C")),
            task="combine evidence",
            max_turns=2,
            final_answer_agent="C",
        )
        self.assertEqual(result.protocol_id, "three-role-card-v1")
        self.assertEqual([turn.speaker for turn in result.turns], ["A", "B"])
        self.assertEqual(agents["A"].calls[0][0]["content"], card.agent_instructions["A"])

    def test_dialogue_card_rejects_bad_schema_duplicate_keys_and_invalid_roles(self) -> None:
        duplicate_role = (
            '{"schema":"tlu.dialogue_protocol_card.v1",'
            '"protocol_id":"x","agent_instructions":{"A":"one","A":"two"}}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DialogueProtocolCard.from_json(duplicate_role)
        for instructions in ({"A": "one"}, {"A": "one", " ": "two"}, {"A": "", "B": "ok"}):
            with self.subTest(instructions=instructions), self.assertRaises(ValueError):
                DialogueProtocolCard("invalid-card", instructions)

    def test_shipped_dialogue_card_uses_the_public_schema(self) -> None:
        card_path = Path(__file__).resolve().parents[1] / "examples" / "dialogue_protocol_card.json"
        card = DialogueProtocolCard.from_json(card_path.read_bytes())
        self.assertEqual(set(card.agent_instructions), {"A", "B", "C"})

    def test_rejects_duplicate_extra_wrong_schema_and_invalid_json(self) -> None:
        duplicate = (
            '{"schema":"tlu.shared_protocol_card.v1",'
            '"protocol_id":"first","protocol_id":"second",'
            '"sender_instruction":"s","receiver_instruction":"r"}'
        )
        for payload in (duplicate, "{", b"\xff"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ProtocolCard.from_json(payload)

        for changes in (
            {"extra": "field"},
            {"schema": "tlu.shared_protocol_card.v0"},
        ):
            data = self.card.to_dict()
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ProtocolCard.from_json(json.dumps(data))

    def test_rejects_invalid_fields_and_size_limits(self) -> None:
        for kwargs in (
            {"protocol_id": "  "},
            {"protocol_id": "x" * 129},
            {"sender_instruction": "\t"},
            {"receiver_instruction": "é" * 16_385},
        ):
            values = {
                "protocol_id": self.card.protocol_id,
                "sender_instruction": self.card.sender_instruction,
                "receiver_instruction": self.card.receiver_instruction,
            }
            values.update(kwargs)
            with self.subTest(field=next(iter(kwargs))), self.assertRaises(ValueError):
                ProtocolCard(**values)


if __name__ == "__main__":
    unittest.main()
