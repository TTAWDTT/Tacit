from __future__ import annotations

import unittest
from unittest.mock import patch
import json
from urllib.request import Request

from tacit import ChatCompletion, OpenAICompatibleClient, exchange_once


class FakeProtocol:
    protocol_id = "test-v1"
    sender_instruction = "encode evidence"
    receiver_instruction = "interpret evidence"


class FakeModel:
    def __init__(self, completion: ChatCompletion) -> None:
        self.completion = completion
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        return self.completion


class TacitRuntimeTests(unittest.TestCase):
    def test_exchange_delivers_the_exact_sender_string_and_counts_utf8(self) -> None:
        payload = "事实: café 🧪\n"
        sender = FakeModel(ChatCompletion(payload, "sender-v1", 11, 7, 0.2))
        receiver = FakeModel(ChatCompletion("结论", "receiver-v2", 24, 2, 0.3))

        result = exchange_once(
            sender,
            receiver,
            protocol=FakeProtocol(),
            sender_context="private sender context",
            receiver_context="private receiver context",
            receiver_task="decide",
        )

        self.assertEqual(result.message, payload)
        self.assertIn(payload, receiver.calls[0][1]["content"])
        self.assertEqual(result.payload_bytes, len(payload.encode("utf-8")))
        self.assertEqual(result.transmission_record(
            round_number=1,
            sender_name="A",
            recipient_name="B",
            transport_boundary="inter_process",
            framing_bytes=3,
            recipient_tokenizer="receiver-tokenizer-v1",
            recipient_payload_tokens=13,
        ), {
            "round": 1,
            "sender": "A",
            "recipients": ["B"],
            "payload_bytes": len(payload.encode("utf-8")),
            "framing_bytes": 3,
            "encoding": "utf-8",
            "media_type": "text/plain; charset=utf-8",
            "transport_boundary": "inter_process",
            "payload_metadata": {"protocol_id": "test-v1"},
            "recipient_tokens": {"B": {"tokenizer": "receiver-tokenizer-v1", "tokens": 13}},
        })

    def test_missing_native_token_usage_stays_missing(self) -> None:
        sender = FakeModel(ChatCompletion("msg", "sender"))
        receiver = FakeModel(ChatCompletion("done", "receiver"))
        result = exchange_once(
            sender,
            receiver,
            protocol=FakeProtocol(),
            sender_context="x",
            receiver_context="y",
            receiver_task="z",
        )
        record = result.transmission_record(
            round_number=1,
            sender_name="A",
            recipient_name="B",
            transport_boundary="network",
        )
        self.assertNotIn("recipient_tokens", record)

    def test_transmission_rejects_invalid_boundary_and_counts(self) -> None:
        result = exchange_once(
            FakeModel(ChatCompletion("m", "a")),
            FakeModel(ChatCompletion("r", "b")),
            protocol=FakeProtocol(),
            sender_context="x",
            receiver_context="y",
            receiver_task="z",
        )
        for kwargs in (
            {"round_number": 0, "transport_boundary": "network"},
            {"round_number": 1.5, "transport_boundary": "network"},
            {"round_number": 1, "transport_boundary": "same_process"},
            {"round_number": 1, "transport_boundary": "network", "framing_bytes": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                result.transmission_record(sender_name="A", recipient_name="B", **kwargs)

    def test_openai_compatible_client_preserves_content_and_usage(self) -> None:
        response_data = {
            "id": "request-1",
            "model": "actual-model",
            "choices": [{"message": {"content": "  wire text\n"}}],
            "usage": {"prompt_tokens": 31, "completion_tokens": 4, "generation_seconds": 0.25},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_data).encode()

        with patch("tacit.runtime.urlopen", return_value=Response()) as open_url:
            completion = OpenAICompatibleClient(
                "http://localhost:8000/v1/", "requested-model", api_key="secret"
            ).complete([{"role": "user", "content": "hello"}])

        request = open_url.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(completion, ChatCompletion(
            "  wire text\n", "actual-model", 31, 4, 0.25, "request-1"
        ))


if __name__ == "__main__":
    unittest.main()
