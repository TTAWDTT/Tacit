from __future__ import annotations

import unittest
from unittest.mock import patch
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request

from tacit import ChatCompletion, LocalTCPMessageChannel, OpenAICompatibleClient, exchange_once
from tools.cost_report import aggregate


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
    def test_loopback_channel_delivers_exact_utf8_and_partitions_wire_body(self) -> None:
        received = []
        message = '事实: "café" 🧪\\line\n'
        with LocalTCPMessageChannel(received.append) as channel:
            transmission = channel.send(
                message,
                protocol_id="json-envelope-test-v1",
                round_number=2,
                sender="A",
                recipient="B",
            )

        self.assertEqual(received[0]["payload"], message)
        self.assertEqual(transmission.message, message)
        self.assertEqual(transmission.logical_payload_bytes, len(message.encode("utf-8")))
        self.assertEqual(
            transmission.total_application_bytes,
            transmission.serialized_payload_bytes + transmission.framing_bytes,
        )
        record = transmission.cost_record()
        self.assertEqual(record["payload_bytes"] + record["framing_bytes"], transmission.total_application_bytes)
        self.assertEqual(record["transport_boundary"], "network")
        self.assertEqual(record["payload_metadata"]["protocol_id"], "json-envelope-test-v1")
        episode = {
            "schema_version": "tlu.costs.v3",
            "episode_id": "loopback-test-episode",
            "stratum": {
                "experiment_id": "loopback-test",
                "task_id": "toy@1",
                "split": "test",
                "task_parameters": {},
                "model_population_id": "fake",
                "agent_models": {"A": "fake-v1", "B": "fake-v1"},
                "scorer_id": "exact-v1",
            },
            "protocol": {
                "policy_id": "fixed",
                "code_id": record["payload_metadata"]["protocol_id"],
                "decoder_id": "test-decoder-v1",
            },
            "outcome": {"joint_success": True, "answer_score": 1.0},
            "transmissions": [record],
            "model_calls": [],
            "runtime": {},
            "setup": [],
        }
        report = aggregate([episode])
        self.assertEqual(report["groups"][0]["channel"]["wire_bytes"]["observed_sum"],
                         transmission.total_application_bytes)

    def test_loopback_channel_surfaces_receiver_callback_failure(self) -> None:
        def reject(_envelope):
            raise ValueError("receiver rejected frame")

        with LocalTCPMessageChannel(reject) as channel:
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                channel.send(
                    "payload",
                    protocol_id="test-v1",
                    round_number=1,
                    sender="A",
                    recipient="B",
                )

    def test_loopback_channel_rejects_invalid_frame_fields(self) -> None:
        with LocalTCPMessageChannel(lambda _envelope: None) as channel:
            for fields in (
                {"message": "x", "protocol_id": "p", "round_number": 0, "sender": "A", "recipient": "B"},
                {"message": 7, "protocol_id": "p", "round_number": 1, "sender": "A", "recipient": "B"},
                {"message": "x", "protocol_id": " ", "round_number": 1, "sender": "A", "recipient": "B"},
            ):
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    channel.send(**fields)

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
            "choices": [{"message": {"content": "  wire text\n"}, "finish_reason": "stop"}],
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
            "  wire text\n", "actual-model", 31, 4, 0.25, "request-1", "stop"
        ))

    def test_loopback_client_can_reject_redirects(self) -> None:
        redirected_requests = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.redirect_port}/capture")
                self.end_headers()

            def log_message(self, *args):
                pass

        class CaptureHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                redirected_requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"content":"leaked"}}]}')

            def log_message(self, *args):
                pass

        capture = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        redirect.redirect_port = capture.server_port
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (capture, redirect)
        ]
        for thread in threads:
            thread.start()
        try:
            client = OpenAICompatibleClient(
                f"http://127.0.0.1:{redirect.server_port}/v1", "local-model",
                follow_redirects=False,
            )
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                client.complete([{"role": "user", "content": "synthetic prompt"}])
            self.assertEqual(redirected_requests, [])
        finally:
            for server in (redirect, capture):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
