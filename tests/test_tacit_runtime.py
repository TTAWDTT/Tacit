from __future__ import annotations

import unittest
from unittest.mock import patch
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request

from tacit import (
    ChatCompletion, LocalTCPFrameChannel, LocalTCPMessageChannel,
    OpenAICompatibleClient, exchange_dialogue, exchange_once,
)
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


class FakeDialogueProtocol:
    protocol_id = "fixed-dialogue-test-v1"
    agent_instructions = {
        "A": "Use A's evidence and the messages visible to you.",
        "B": "Use B's evidence and the messages visible to you.",
    }


class ThreeAgentDialogueProtocol:
    protocol_id = "fixed-three-agent-test-v1"
    agent_instructions = {
        "A": "Use only your local context and messages routed to you.",
        "B": "Use only your local context and messages routed to you.",
        "C": "Use only your local context and messages routed to you.",
    }


class SequencedDialogueModel:
    def __init__(self, model_name, texts):
        self.model_name = model_name
        self.texts = list(texts)
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        text = self.texts.pop(0)
        return ChatCompletion(text, self.model_name, 20, 4, 0.01)


class TacitRuntimeTests(unittest.TestCase):
    def test_fixed_schedule_dialogue_preserves_private_views_and_measures_each_turn(self) -> None:
        agent_a = SequencedDialogueModel("model-a", ["A first", "A follow-up"])
        agent_b = SequencedDialogueModel("model-b", ["B reply"])
        result = exchange_dialogue(
            {"A": agent_a, "B": agent_b},
            protocol=FakeDialogueProtocol(),
            private_contexts={"A": "A-secret", "B": "B-secret"},
            schedule=("A", "B", "A"),
            task="Combine the two observations.",
            max_turns=3,
        )

        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.stop_reason, "schedule_complete")
        self.assertEqual([turn.speaker for turn in result.turns], ["A", "B", "A"])
        self.assertEqual([turn.recipient for turn in result.turns], ["B", "A", "B"])
        self.assertEqual(result.model_call_records()[0]["tokenizer"], "not_reported")
        with self.assertRaisesRegex(ValueError, "tokenizers must map"):
            result.model_call_records(tokenizers={"A": ""})
        self.assertEqual(result.wire_bytes, sum(
            record["payload_bytes"] + record["framing_bytes"]
            for record in result.transmission_records()
        ))
        call_records = result.model_call_records(tokenizers={"A": "tokenizer-a", "B": "tokenizer-b"})
        self.assertEqual([row["agent"] for row in call_records], ["A", "B", "A"])
        episode = {
            "schema_version": "tlu.costs.v3",
            "episode_id": "fixed-dialogue-test-episode",
            "stratum": {
                "experiment_id": "dialogue-runtime-test",
                "task_id": "toy-dialogue@1",
                "split": "test",
                "task_parameters": {"turns": 3},
                "model_population_id": "fake-pair",
                "agent_models": {"A": "model-a", "B": "model-b"},
                "scorer_id": "test-only",
            },
            "protocol": {
                "policy_id": "fixed-schedule",
                "code_id": result.protocol_id,
                "decoder_id": "verbatim-transcript-v1",
            },
            "outcome": {"joint_success": True, "answer_score": 1.0},
            "transmissions": result.transmission_records(),
            "model_calls": call_records,
            "runtime": {},
            "setup": [],
        }
        report = aggregate([episode])
        self.assertEqual(
            report["groups"][0]["channel"]["wire_bytes"]["observed_sum"],
            result.wire_bytes,
        )

        first_a = json.loads(agent_a.calls[0][1]["content"])
        b_prompt = json.loads(agent_b.calls[0][1]["content"])
        second_a = json.loads(agent_a.calls[1][1]["content"])
        self.assertEqual(first_a["private_context"], "A-secret")
        self.assertEqual(first_a["visible_transcript"], [])
        self.assertEqual(b_prompt["private_context"], "B-secret")
        self.assertEqual(b_prompt["visible_transcript"], [{"sender": "A", "message": "A first"}])
        self.assertEqual(second_a["private_context"], "A-secret")
        self.assertEqual(second_a["visible_transcript"], [
            {"sender": "A", "message": "A first"},
            {"sender": "B", "message": "B reply"},
        ])
        self.assertNotIn("B-secret", json.dumps(agent_a.calls))
        self.assertNotIn("A-secret", json.dumps(agent_b.calls))

    def test_multi_agent_schedule_routes_unicast_and_isolates_transcripts(self) -> None:
        agents = {
            "A": SequencedDialogueModel("model-a", ["A to B", "A to C"]),
            "B": SequencedDialogueModel("model-b", ["B to A"]),
            "C": SequencedDialogueModel("model-c", ["C to A", "C to B"]),
        }
        result = exchange_dialogue(
            agents,
            protocol=ThreeAgentDialogueProtocol(),
            private_contexts={"A": "A-secret", "B": "B-secret", "C": "C-secret"},
            schedule=(("A", "B"), ("C", "A"), ("A", "C"), ("C", "B"), ("B", "A")),
            task="Reconcile only the evidence received by each agent.",
            max_turns=5,
        )

        self.assertEqual(result.stop_reason, "schedule_complete")
        self.assertEqual(
            [(turn.speaker, turn.recipient) for turn in result.turns],
            [("A", "B"), ("C", "A"), ("A", "C"), ("C", "B"), ("B", "A")],
        )
        self.assertEqual(
            [record["recipients"] for record in result.transmission_records()],
            [["B"], ["A"], ["C"], ["B"], ["A"]],
        )
        a_second_turn = json.loads(agents["A"].calls[1][1]["content"])
        self.assertEqual(a_second_turn["visible_transcript"], [
            {"sender": "A", "message": "A to B"},
            {"sender": "C", "message": "C to A"},
        ])
        b_first_turn = json.loads(agents["B"].calls[0][1]["content"])
        self.assertEqual(b_first_turn["visible_transcript"], [
            {"sender": "A", "message": "A to B"},
            {"sender": "C", "message": "C to B"},
        ])
        c_first_turn = json.loads(agents["C"].calls[0][1]["content"])
        c_second_turn = json.loads(agents["C"].calls[1][1]["content"])
        self.assertEqual(c_first_turn["visible_transcript"], [])
        self.assertEqual(c_second_turn["visible_transcript"], [
            {"sender": "C", "message": "C to A"},
            {"sender": "A", "message": "A to C"},
        ])
        self.assertNotIn("A to B", json.dumps(agents["C"].calls))
        self.assertNotIn("B-secret", json.dumps(agents["A"].calls))

    def test_final_answer_is_sealed_and_counted_as_inference_not_communication(self) -> None:
        agent_a = SequencedDialogueModel("model-a", ["evidence to B"])
        agent_b = SequencedDialogueModel("model-b", ["candidate-7"])
        result = exchange_dialogue(
            {"A": agent_a, "B": agent_b},
            protocol=FakeDialogueProtocol(),
            private_contexts={"A": "A-secret", "B": "B-secret"},
            schedule=("A",),
            task="Return one candidate ID.",
            max_turns=1,
            final_answer_agent="B",
            final_answer_instruction="Return only the exact candidate ID.",
        )

        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.final_agent, "B")
        self.assertEqual(result.final_submission.text, "candidate-7")
        self.assertEqual(len(result.transmission_records()), 1)
        self.assertEqual(result.wire_bytes, sum(
            row["payload_bytes"] + row["framing_bytes"]
            for row in result.transmission_records()
        ))
        self.assertEqual(
            [row["stage"] for row in result.model_call_records()],
            ["dialogue_turn", "final_answer"],
        )
        final_prompt = json.loads(agent_b.calls[0][1]["content"])
        self.assertEqual(final_prompt["visible_transcript"], [
            {"sender": "A", "message": "evidence to B"},
        ])
        self.assertNotIn("candidate-7", json.dumps(result.transmission_records()))

    def test_dialogue_enforces_total_wire_budget_and_keeps_rejected_call_costs(self) -> None:
        protocol = FakeDialogueProtocol()
        schedule = ("A", "B", "A")
        messages = ("A first", "B reply", "A follow-up")
        with LocalTCPMessageChannel(lambda _envelope: None) as channel:
            first = channel.measure(
                messages[0], protocol_id=protocol.protocol_id, round_number=1,
                sender="A", recipient="B",
            )
            second = channel.measure(
                messages[1], protocol_id=protocol.protocol_id, round_number=2,
                sender="B", recipient="A",
            )
        budget = first.total_application_bytes + second.total_application_bytes - 1
        agent_a = SequencedDialogueModel("model-a", [messages[0], messages[2]])
        agent_b = SequencedDialogueModel("model-b", [messages[1]])

        result = exchange_dialogue(
            {"A": agent_a, "B": agent_b},
            protocol=protocol,
            private_contexts={"A": "A-secret", "B": "B-secret"},
            schedule=schedule,
            task="Combine the two observations.",
            max_turns=3,
            wire_budget_bytes=budget,
        )

        self.assertEqual(result.stop_reason, "wire_budget_exhausted")
        self.assertEqual(result.wire_budget_bytes, budget)
        self.assertLessEqual(result.wire_bytes, budget)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual([turn.round_number for turn in result.turns], [1, 2])
        self.assertIsNotNone(result.turns[0].transmission)
        self.assertIsNone(result.turns[1].transmission)
        self.assertEqual(len(result.transmission_records()), 1)
        self.assertEqual(len(result.model_call_records()), 2)
        self.assertEqual([row["round"] for row in result.model_call_records()], [1, 2])
        self.assertEqual(len(agent_a.calls), 1)
        self.assertEqual(len(agent_b.calls), 1)

    def test_dialogue_rejects_invalid_wire_budget_before_model_calls(self) -> None:
        agents = {
            "A": FakeModel(ChatCompletion("a", "A")),
            "B": FakeModel(ChatCompletion("b", "B")),
        }
        for invalid_budget in (-1, 1.5, True):
            with self.subTest(budget=invalid_budget), self.assertRaisesRegex(
                ValueError, "wire_budget_bytes"
            ):
                exchange_dialogue(
                    agents,
                    protocol=FakeDialogueProtocol(),
                    private_contexts={"A": "a", "B": "b"},
                    schedule=("A",),
                    task="task",
                    max_turns=1,
                    wire_budget_bytes=invalid_budget,
                )
        self.assertEqual(agents["A"].calls, [])
        self.assertEqual(agents["B"].calls, [])

    def test_dialogue_does_not_call_models_when_no_message_can_fit(self) -> None:
        agents = {
            "A": FakeModel(ChatCompletion("a", "A")),
            "B": FakeModel(ChatCompletion("b", "B")),
        }
        result = exchange_dialogue(
            agents,
            protocol=FakeDialogueProtocol(),
            private_contexts={"A": "a", "B": "b"},
            schedule=("A", "B"),
            task="task",
            max_turns=2,
            wire_budget_bytes=0,
        )
        self.assertEqual(result.stop_reason, "wire_budget_exhausted")
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(result.wire_bytes, 0)
        self.assertEqual(agents["A"].calls, [])
        self.assertEqual(agents["B"].calls, [])

    def test_dialogue_rejects_invalid_schedules_and_roles(self) -> None:
        agents = {"A": FakeModel(ChatCompletion("a", "A")), "B": FakeModel(ChatCompletion("b", "B"))}
        common = {
            "agents": agents,
            "protocol": FakeDialogueProtocol(),
            "private_contexts": {"A": "a", "B": "b"},
            "task": "task",
            "max_turns": 2,
        }
        for changes in (
            {"schedule": ()},
            {"schedule": ("A", "C")},
            {"schedule": ("A", [])},
            {"schedule": ("A", "B", "A")},
            {"private_contexts": {"A": "a"}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                exchange_dialogue(**(common | {"schedule": ("A", "B")} | changes))

    def test_empty_schedule_is_valid_only_for_sealed_final_answer(self) -> None:
        agents = {"A": FakeModel(ChatCompletion("a", "A")), "B": FakeModel(ChatCompletion("b", "B"))}
        result = exchange_dialogue(
            agents, protocol=FakeDialogueProtocol(), private_contexts={"A": "a", "B": "b"},
            schedule=(), task="task", max_turns=0, final_answer_agent="B",
        )
        self.assertEqual(result.turns, ())
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.transmission_records(), [])

    def test_multi_agent_dialogue_rejects_implicit_or_invalid_routes_before_calls(self) -> None:
        agents = {
            name: FakeModel(ChatCompletion(name, f"model-{name}"))
            for name in ("A", "B", "C")
        }
        common = {
            "agents": agents,
            "protocol": ThreeAgentDialogueProtocol(),
            "private_contexts": {"A": "a", "B": "b", "C": "c"},
            "task": "task",
            "max_turns": 1,
        }
        for schedule in (("A",), (("A", "A"),), (("A", "Z"),), (("A", "B", "C"),)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                exchange_dialogue(**common, schedule=schedule)
        self.assertTrue(all(not agent.calls for agent in agents.values()))
        with self.assertRaisesRegex(ValueError, "final_answer_agent"):
            exchange_dialogue(**common, schedule=(("A", "B"),), final_answer_agent="Z")
        self.assertTrue(all(not agent.calls for agent in agents.values()))

    def test_loopback_frame_channel_transmits_opaque_bytes_and_accounts_exactly(self) -> None:
        received = []
        payload = b"\x00\xffKV\x80\x00"
        with LocalTCPFrameChannel(lambda metadata, body: received.append((metadata, body))) as channel:
            transmission = channel.send(
                payload,
                protocol_id="opaque-repr-v1",
                round_number=3,
                sender="A",
                recipient="B",
                media_type="application/vnd.tlu.tensor",
                encoding="float16-le",
                payload_metadata={"dtype": "float16", "shape": [1, 3]},
            )

        self.assertEqual(received[0][1], payload)
        self.assertEqual(received[0][0]["payload_length"], len(payload))
        self.assertEqual(transmission.payload_bytes, len(payload))
        self.assertEqual(
            transmission.total_application_bytes,
            transmission.payload_bytes + transmission.framing_bytes,
        )
        record = transmission.cost_record()
        self.assertEqual(record["payload_bytes"], len(payload))
        self.assertEqual(record["media_type"], "application/vnd.tlu.tensor")
        self.assertEqual(record["encoding"], "float16-le")
        self.assertEqual(record["payload_metadata"]["shape"], [1, 3])
        self.assertEqual(record["payload_bytes"] + record["framing_bytes"], transmission.total_application_bytes)
        episode = {
            "schema_version": "tlu.costs.v3",
            "episode_id": "binary-loopback-test",
            "stratum": {
                "experiment_id": "frame-channel-test", "task_id": "toy@1", "split": "test",
                "task_parameters": {}, "model_population_id": "fake",
                "agent_models": {"A": "fake-v1", "B": "fake-v1"}, "scorer_id": "none",
            },
            "protocol": {"policy_id": "fixed", "code_id": "opaque-repr-v1", "decoder_id": "none"},
            "outcome": {"joint_success": True, "answer_score": 1.0},
            "transmissions": [record], "model_calls": [], "runtime": {}, "setup": [],
        }
        report = aggregate([episode])
        self.assertEqual(report["groups"][0]["channel"]["wire_bytes"]["observed_sum"],
                         transmission.total_application_bytes)

    def test_frame_channel_rejects_invalid_metadata_and_callback_errors(self) -> None:
        with LocalTCPFrameChannel(lambda _metadata, _payload: None, max_frame_bytes=128) as channel:
            with self.assertRaisesRegex(ValueError, "JSON values"):
                channel.send(b"x", protocol_id="p", round_number=1, sender="A", recipient="B",
                             payload_metadata={"unsupported": object()})
            with self.assertRaisesRegex(ValueError, "exceeds"):
                channel.send(b"x" * 128, protocol_id="p", round_number=1, sender="A", recipient="B")

        def reject(_metadata, _payload):
            raise ValueError("rejected")

        with LocalTCPFrameChannel(reject) as channel:
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                channel.send(b"opaque", protocol_id="p", round_number=1, sender="A", recipient="B")

    def test_loopback_channel_delivers_exact_utf8_and_partitions_wire_body(self) -> None:
        received = []
        message = '事实: "café" 🧪\\line\n'
        with LocalTCPMessageChannel(received.append) as channel:
            measured = channel.measure(
                message, protocol_id="json-envelope-test-v1", round_number=2,
                sender="A", recipient="B",
            )
            transmission = channel.send(
                message,
                protocol_id="json-envelope-test-v1",
                round_number=2,
                sender="A",
                recipient="B",
            )

        self.assertEqual(received[0]["payload"], message)
        self.assertEqual(measured, transmission)
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

    def test_openai_compatible_client_keeps_reasoning_metadata_separate_and_private(self) -> None:
        response_data = {
            "model": "local-qwen",
            "choices": [{
                "message": {
                    "content": "candidate-7",
                    "reasoning_content": "private model reasoning must not enter the wire message",
                },
                "finish_reason": "stop",
            }],
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_data).encode()

        with patch("tacit.runtime.urlopen", return_value=Response()):
            completion = OpenAICompatibleClient(
                "http://localhost:8000/v1", "local-qwen"
            ).complete([{"role": "user", "content": "return a candidate ID"}])

        self.assertEqual(completion.text, "candidate-7")
        self.assertTrue(completion.reasoning_content_present)
        self.assertNotIn("private model reasoning", repr(completion))

    def test_openai_compatible_client_rejects_malformed_reasoning_metadata(self) -> None:
        response_data = {
            "model": "local-qwen",
            "choices": [{
                "message": {"content": "candidate-7", "reasoning_content": {"text": "x"}}
            }],
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_data).encode()

        with patch("tacit.runtime.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "reasoning_content must be text or null"):
                OpenAICompatibleClient(
                    "http://localhost:8000/v1", "local-qwen"
                ).complete([{"role": "user", "content": "return a candidate ID"}])

    def test_openai_compatible_client_sends_frozen_temperature_when_configured(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"model": "m", "choices": [{"message": {"content": "ok"}}]}).encode()

        with patch("tacit.runtime.urlopen", return_value=Response()) as open_url:
            OpenAICompatibleClient("http://localhost:8000/v1", "m", temperature=0.0).complete(
                [{"role": "user", "content": "hello"}]
            )
        request_body = json.loads(open_url.call_args.args[0].data)
        self.assertEqual(request_body["temperature"], 0.0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            OpenAICompatibleClient("http://localhost:8000/v1", "m", temperature=float("nan")).complete([])

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
