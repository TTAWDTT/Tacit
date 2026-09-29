from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.private_match_v0_1.generate_tasks import generate_episode
from experiments.private_match_v0_2.protocols import PROTOCOL_IDS, protocol_by_id
from experiments.private_match_v0_2.report import private_match_report
from experiments.private_match_v0_2.runner import _message_diagnostics, run_condition
from tacit.runtime import ChatCompletion
from tools.cost_report import read_jsonl


class FakeModel:
    def __init__(self, name: str, responder):
        self.model_name = name
        self.responder = responder

    def complete(self, messages):
        return ChatCompletion(
            text=self.responder(messages), model=self.model_name,
            input_tokens=12, output_tokens=3, service_seconds=0.01,
            finish_reason="stop",
        )


class PrivateMatchV02Tests(unittest.TestCase):
    def test_registered_protocols_are_explicit_and_unique(self):
        protocols = [protocol_by_id(name) for name in PROTOCOL_IDS]
        self.assertEqual(len({item.protocol_id for item in protocols}), len(protocols))
        self.assertTrue(all(item.sender_instruction and item.receiver_instruction for item in protocols))

    def test_hex_nibble_mapping_is_exact_for_registered_domain(self):
        for value in range(16):
            code = format(value, "x")
            self.assertEqual(int(code, 16), value)

    def test_message_fidelity_diagnostics_separate_syntax_from_meaning(self):
        fields = ["f0", "f1"]
        target = {"f0": "v0002", "f1": "v0010"}
        self.assertEqual(_message_diagnostics("json", '{"f0":"v0002","f1":"v0010"}', target, fields, 16), (True, True))
        self.assertEqual(_message_diagnostics("json", '{"f0":"v0002","f1":"v0009"}', target, fields, 16), (True, False))
        self.assertEqual(_message_diagnostics("json", '{"f0":"v0002"}', target, fields, 16), (False, False))
        self.assertEqual(_message_diagnostics("compact_kv", "f0=v0002;f1=v0010", target, fields, 16), (True, True))
        self.assertEqual(_message_diagnostics("compact_kv", "f0=v0002;f1=v0009", target, fields, 16), (True, False))
        self.assertEqual(_message_diagnostics("compact_kv", "f1=v0010;f0=v0002", target, fields, 16), (False, False))
        self.assertEqual(_message_diagnostics("tuple", "v0002,v0010", target, fields, 16), (True, True))
        self.assertEqual(_message_diagnostics("tuple", "v0002, v0010", target, fields, 16), (False, False))
        self.assertEqual(_message_diagnostics("hex_nibbles", "2a", target, fields, 16), (True, True))
        self.assertEqual(_message_diagnostics("hex_nibbles", "2z", target, fields, 16), (False, False))
        self.assertEqual(_message_diagnostics("concise_nl", "record: ...", target, fields, 16), (None, None))

    def test_message_condition_uses_real_channel_and_emits_valid_v3_record(self):
        sender_view, receiver_view, gold = generate_episode(
            episode_id="pm2-test", seed=92, candidate_count=8,
            feature_count=5, vocabulary_size=16,
        )
        match_id = gold["target_candidate_id"]
        target_record = sender_view["target_record"]

        def sender(messages):
            return json.dumps(target_record, separators=(",", ":"))

        def receiver(messages):
            payload = json.loads(messages[1]["content"])
            self.assertEqual(payload["message_verbatim"], json.dumps(target_record, separators=(",", ":")))
            return match_id

        row = run_condition(
            episode_id="pm2-test", seed=92, candidate_count=8,
            feature_count=5, vocabulary_size=16, protocol=protocol_by_id("json"),
            sender_model=FakeModel("fake-sender", sender),
            receiver_model=FakeModel("fake-receiver", receiver),
            sender_tokenizer_id="fake-sender-tokenizer-v1",
            receiver_tokenizer_id="fake-receiver-tokenizer-v1", model_population_id="fake-population",
        )
        self.assertTrue(row["outcome"]["joint_success"])
        self.assertNotIn("seed", row["stratum"]["task_parameters"])
        self.assertEqual(row["diagnostics"]["generation_seed"], 92)
        self.assertEqual(row["model_calls"][0]["tokenizer"], "fake-sender-tokenizer-v1")
        self.assertEqual(row["model_calls"][1]["tokenizer"], "fake-receiver-tokenizer-v1")
        self.assertEqual(len(row["transmissions"]), 1)
        tx = row["transmissions"][0]
        self.assertEqual(tx["transport_boundary"], "network")
        self.assertEqual(tx["payload_metadata"]["application_layer_scope"], "length-prefixed loopback TCP; TCP/IP headers excluded")
        self.assertIsNone(tx["recipient_tokens"]["receiver"]["tokens"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(len(read_jsonl(path)), 1)

    def test_no_message_has_no_transmission_and_one_receiver_call(self):
        def receiver(messages):
            return "r0000"

        row = run_condition(
            episode_id="pm2-empty", seed=5, candidate_count=8,
            feature_count=5, vocabulary_size=16, protocol=protocol_by_id("concise_nl"),
            sender_model=None, receiver_model=FakeModel("fake-receiver", receiver),
            sender_tokenizer_id=None,
            receiver_tokenizer_id="fake-receiver-tokenizer-v1", model_population_id="fake-population",
        )
        self.assertEqual(row["transmissions"], [])
        self.assertEqual(len(row["model_calls"]), 1)
        self.assertEqual(row["protocol"]["policy_id"], "no_message")
        self.assertIsNone(row["diagnostics"]["message_text"])
        self.assertIsNone(row["diagnostics"]["sender_truncated"])

    def test_report_separates_fidelity_from_success_and_pairs_by_episode(self):
        rows = []
        for index, seed in enumerate((111, 222)):
            episode_id = f"report-{index}"
            sender_view, receiver_view, gold = generate_episode(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16,
            )
            target = sender_view["target_record"]
            correct_id = gold["target_candidate_id"]

            no_message = run_condition(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16, protocol=protocol_by_id("no_message"),
                sender_model=None, receiver_model=FakeModel("receiver", lambda _: "bad-id"),
                sender_tokenizer_id=None, receiver_tokenizer_id="receiver-tok",
                model_population_id="pair-v1",
            )
            json_row = run_condition(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16, protocol=protocol_by_id("json"),
                sender_model=FakeModel("sender", lambda _: json.dumps(target, separators=(",", ":"))),
                receiver_model=FakeModel("receiver", lambda messages: correct_id),
                sender_tokenizer_id="sender-tok", receiver_tokenizer_id="receiver-tok",
                model_population_id="pair-v1",
            )
            rows.extend((no_message, json_row))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = private_match_report(read_jsonl(path), replicates=100)

        json_condition = next(item for item in report["conditions"] if item["protocol"]["policy_id"] == "json")
        self.assertEqual(json_condition["episodes"], 2)
        self.assertEqual(json_condition["diagnostics"]["message_semantic_fidelity"]["rate"], 1.0)
        self.assertEqual(json_condition["task"]["joint_success_rate"], 1.0)
        comparison = next(item for item in report["paired_comparisons"] if item["paired_episode_count"] == 2)
        self.assertFalse(comparison["model_strata_matched"])
        self.assertEqual(comparison["metrics"]["joint_success"]["mean_left_minus_right"], 1.0)
        self.assertEqual(comparison["metrics"]["message_semantic_fidelity"]["paired_episodes"], 0)
        self.assertEqual(comparison["metrics"]["message_semantic_fidelity"]["missing_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
