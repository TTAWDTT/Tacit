from __future__ import annotations

import copy
from itertools import combinations, product
import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.private_match_v0_1.generate_tasks import generate_episode
from experiments.private_match_v0_2.protocols import PROTOCOL_IDS, protocol_by_id
from experiments.private_match_v0_2.code_bounds import bit_budget_frontier, optimal_success_probability
from experiments.private_match_v0_2.report import main as report_main, private_match_report
from experiments.private_match_v0_2.runner import (
    CAPABILITY_CONDITION, MAX_MODEL_CALLS_PER_BATCH, _loopback_url, _message_diagnostics,
    episode_id_for_seed, main as runner_main, planned_model_calls, run_condition,
    validate_capability_ledger,
)
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
    def test_batch_call_budget_and_seed_stable_episode_ids(self):
        self.assertEqual(MAX_MODEL_CALLS_PER_BATCH, 12)
        self.assertEqual(planned_model_calls(4, ["no_message", "concise_nl"]), 12)
        self.assertEqual(planned_model_calls(4, ["no_message", *PROTOCOL_IDS]), 52)
        self.assertEqual(planned_model_calls(4, ["full_information", "no_message", "concise_nl"]), 16)
        self.assertEqual(episode_id_for_seed(20260929), "pm2-000020260929")
        self.assertEqual(episode_id_for_seed(20260929), episode_id_for_seed(20260929))

    def test_registered_protocols_are_explicit_and_unique(self):
        protocols = [protocol_by_id(name) for name in PROTOCOL_IDS]
        self.assertEqual(len({item.protocol_id for item in protocols}), len(protocols))
        self.assertTrue(all(item.sender_instruction and item.receiver_instruction for item in protocols))

    def test_endpoint_guard_accepts_loopback_and_rejects_remote_hosts(self):
        self.assertEqual(_loopback_url("http://127.0.0.1:8001/v1"), "http://127.0.0.1:8001/v1")
        self.assertEqual(_loopback_url("http://localhost:8002/v1"), "http://localhost:8002/v1")
        for endpoint in ("https://api.example.com/v1", "http://192.168.1.4:8000/v1"):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "loopback"):
                _loopback_url(endpoint)

    def test_execute_rejects_missing_resource_preflight_before_output_or_client_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "must-not-exist.jsonl"
            arguments = [
                "runner.py", "--execute", "--protocols", "full_information", "--episodes", "1",
                "--receiver-model", "local-receiver", "--receiver-tokenizer-id", "receiver-tokenizer",
                "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                with patch("experiments.private_match_v0_2.runner.OpenAICompatibleClient") as client:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        runner_main()
            self.assertFalse(output_path.exists())
            client.assert_not_called()

    def test_execute_rejects_protocol_batch_without_capability_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "must-not-exist.jsonl"
            arguments = [
                "runner.py", "--execute", "--protocols", "no_message", "--episodes", "1",
                "--receiver-model", "local-receiver", "--receiver-tokenizer-id", "receiver-tokenizer",
                "--resource-preflight", str(Path(directory) / "preflight.json"),
                "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                with patch("experiments.private_match_v0_2.runner.OpenAICompatibleClient") as client:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        runner_main()
            self.assertFalse(output_path.exists())
            client.assert_not_called()

    def test_execute_rejects_mixed_capability_and_comparison_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "must-not-exist.jsonl"
            arguments = [
                "runner.py", "--execute", "--protocols", "full_information", "no_message", "--episodes", "1",
                "--receiver-model", "local-receiver", "--receiver-tokenizer-id", "receiver-tokenizer",
                "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                with patch("experiments.private_match_v0_2.runner.OpenAICompatibleClient") as client:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        runner_main()
            self.assertFalse(output_path.exists())
            client.assert_not_called()

    def test_execute_rejects_duplicate_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "must-not-exist.jsonl"
            arguments = [
                "runner.py", "--execute", "--protocols", "full_information", "full_information", "--episodes", "1",
                "--receiver-model", "local-receiver", "--receiver-tokenizer-id", "receiver-tokenizer",
                "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                with patch("experiments.private_match_v0_2.runner.OpenAICompatibleClient") as client:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        runner_main()
            self.assertFalse(output_path.exists())
            client.assert_not_called()

    def test_execute_checks_every_configured_endpoint_port(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.jsonl"
            arguments = [
                "runner.py", "--execute", "--protocols", "concise_nl", "--episodes", "1",
                "--sender-model", "local-sender", "--sender-tokenizer-id", "sender-tokenizer",
                "--receiver-model", "local-receiver", "--receiver-tokenizer-id", "receiver-tokenizer",
                "--sender-base-url", "http://127.0.0.1:8001/v1",
                "--receiver-base-url", "http://127.0.0.1:8002/v1",
                "--resource-preflight", str(Path(directory) / "preflight.json"),
                "--capability-ledger", str(Path(directory) / "capability.jsonl"),
                "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                with patch("experiments.private_match_v0_2.runner.validate_capability_ledger"):
                    with patch("experiments.private_match_v0_2.runner.validate_resource_preflight") as validate:
                        with patch("experiments.private_match_v0_2.runner.OpenAICompatibleClient"):
                            with patch("experiments.private_match_v0_2.runner.run_condition", return_value={"fake": True}):
                                with redirect_stdout(io.StringIO()):
                                    self.assertEqual(runner_main(), 0)
            validate.assert_called_once_with(Path(directory) / "preflight.json", required_ports={8001, 8002})
            self.assertTrue(output_path.exists())

    def test_full_information_condition_is_unmessaged_and_exact(self):
        episode_id = episode_id_for_seed(123)
        sender_view, receiver_view, gold = generate_episode(
            episode_id=episode_id, seed=123, candidate_count=8,
            feature_count=5, vocabulary_size=16,
        )

        def answer(messages):
            payload = json.loads(messages[1]["content"])
            self.assertEqual(payload["target_record"], sender_view["target_record"])
            self.assertEqual(payload["candidates"], receiver_view["candidates"])
            return gold["target_candidate_id"]

        row = run_condition(
            episode_id=episode_id, seed=123, candidate_count=8, feature_count=5,
            vocabulary_size=16, protocol=protocol_by_id("no_message"), sender_model=None,
            receiver_model=FakeModel("fake-receiver", answer), sender_tokenizer_id=None,
            receiver_tokenizer_id="fake-receiver-tokenizer-v1", model_population_id="fake-population",
            condition_name=CAPABILITY_CONDITION,
        )
        self.assertTrue(row["outcome"]["joint_success"])
        self.assertEqual(row["protocol"]["policy_id"], "full_information_capability_control")
        self.assertEqual(row["transmissions"], [])
        self.assertEqual(len(row["model_calls"]), 1)
        self.assertEqual(row["model_calls"][0]["stage"], CAPABILITY_CONDITION)
        self.assertTrue(row["diagnostics"]["answer_format_valid"])

        whitespace_row = run_condition(
            episode_id=episode_id, seed=123, candidate_count=8, feature_count=5,
            vocabulary_size=16, protocol=protocol_by_id("no_message"), sender_model=None,
            receiver_model=FakeModel("fake-receiver", lambda _messages: f" {gold['target_candidate_id']}\n"),
            sender_tokenizer_id=None, receiver_tokenizer_id="fake-receiver-tokenizer-v1",
            model_population_id="fake-population", condition_name=CAPABILITY_CONDITION,
        )
        self.assertTrue(whitespace_row["outcome"]["joint_success"])
        self.assertFalse(whitespace_row["diagnostics"]["answer_format_valid"])

    def test_capability_ledger_requires_all_exact_matching_episodes_and_models(self):
        rows = []
        for seed in range(200, 204):
            episode_id = episode_id_for_seed(seed)
            sender_view, receiver_view, gold = generate_episode(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16,
            )
            receiver = FakeModel("receiver-v1", lambda _messages, answer=gold["target_candidate_id"]: answer)
            rows.append(run_condition(
                episode_id=episode_id, seed=seed, candidate_count=8, feature_count=5,
                vocabulary_size=16, protocol=protocol_by_id("no_message"), sender_model=None,
                receiver_model=receiver, sender_tokenizer_id=None,
                receiver_tokenizer_id="receiver-tok-v1", model_population_id="population-v1",
                condition_name=CAPABILITY_CONDITION,
            ))

        def validate(rows_to_check, **overrides):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "capability.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows_to_check), encoding="utf-8")
                parameters = {
                    "episodes": 4, "seed": 200, "candidate_count": 8, "feature_count": 5,
                    "vocabulary_size": 16, "receiver_model": "receiver-v1",
                    "receiver_tokenizer_id": "receiver-tok-v1", "model_population_id": "population-v1",
                }
                parameters.update(overrides)
                return validate_capability_ledger(path, **parameters)

        validate(rows)
        failed = copy.deepcopy(rows)
        failed[2]["outcome"]["joint_success"] = False
        with self.assertRaisesRegex(ValueError, "did not pass every episode"):
            validate(failed)
        malformed = copy.deepcopy(rows)
        malformed[1]["diagnostics"]["answer_format_valid"] = False
        with self.assertRaisesRegex(ValueError, "strict valid"):
            validate(malformed)
        with self.assertRaisesRegex(ValueError, "model differs"):
            validate(rows, receiver_model="other-model")
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            validate(rows, receiver_tokenizer_id="other-tokenizer")

    def test_private_match_average_case_frontier_matches_exhaustive_optimum(self):
        for candidate_count in (2, 3, 4):
            for message_count in (1, 2, 3):
                best = 0
                for encoding in product(range(message_count), repeat=4):
                    total = 0
                    set_count = 0
                    for candidates in combinations(range(4), candidate_count):
                        set_count += 1
                        for target in candidates:
                            collision_class_size = sum(encoding[item] == encoding[target] for item in candidates)
                            total += 1 / collision_class_size
                    best = max(best, total / (set_count * candidate_count))
                exact = optimal_success_probability(
                    space_size=4, candidate_count=candidate_count, message_count=message_count,
                )
                self.assertAlmostEqual(float(exact), best)

    def test_private_match_frontier_endpoints_and_monotonicity(self):
        frontier = bit_budget_frontier(space_size=16 ** 5, candidate_count=8, max_bits=20)
        probabilities = [entry["success_probability"] for entry in frontier]
        self.assertEqual(frontier[0]["success_fraction"], "1/8")
        self.assertEqual(frontier[-1]["success_fraction"], "1/1")
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertTrue(all(entry["available_messages"] == min(2 ** entry["payload_bits"], 16 ** 5) for entry in frontier))
        with self.assertRaises(ValueError):
            optimal_success_probability(space_size=8, candidate_count=9, message_count=2)

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
            baseline_path = Path(directory) / "baseline.jsonl"
            message_path = Path(directory) / "message.jsonl"
            baseline_path.write_text("".join(json.dumps(rows[index]) + "\n" for index in range(0, len(rows), 2)), encoding="utf-8")
            message_path.write_text("".join(json.dumps(rows[index]) + "\n" for index in range(1, len(rows), 2)), encoding="utf-8")
            report = private_match_report(
                read_jsonl(baseline_path) + read_jsonl(message_path), replicates=100,
            )

        json_condition = next(item for item in report["conditions"] if item["protocol"]["policy_id"] == "json")
        self.assertEqual(json_condition["episodes"], 2)
        self.assertEqual(json_condition["diagnostics"]["message_semantic_fidelity"]["rate"], 1.0)
        self.assertEqual(json_condition["task"]["joint_success_rate"], 1.0)
        comparison = next(item for item in report["paired_comparisons"] if item["paired_episode_count"] == 2)
        self.assertFalse(comparison["model_strata_matched"])
        self.assertEqual(comparison["metrics"]["joint_success"]["mean_left_minus_right"], 1.0)
        self.assertEqual(comparison["metrics"]["message_semantic_fidelity"]["paired_episodes"], 0)
        self.assertEqual(comparison["metrics"]["message_semantic_fidelity"]["missing_pairs"], 2)

    def test_report_cli_combines_staged_ledgers(self):
        rows = []
        for seed in (71, 72):
            episode_id = episode_id_for_seed(seed)
            sender_view, _, gold = generate_episode(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16,
            )
            rows.append(run_condition(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16, protocol=protocol_by_id("no_message"),
                sender_model=None, receiver_model=FakeModel("receiver", lambda _: "bad"),
                sender_tokenizer_id=None, receiver_tokenizer_id="receiver-tok",
                model_population_id="pair-v1",
            ))
            target = sender_view["target_record"]
            rows.append(run_condition(
                episode_id=episode_id, seed=seed, candidate_count=8,
                feature_count=5, vocabulary_size=16, protocol=protocol_by_id("json"),
                sender_model=FakeModel("sender", lambda _, target=target: json.dumps(target)),
                receiver_model=FakeModel("receiver", lambda _: gold["target_candidate_id"]),
                sender_tokenizer_id="sender-tok", receiver_tokenizer_id="receiver-tok",
                model_population_id="pair-v1",
            ))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            first.write_text("".join(json.dumps(row) + "\n" for row in rows[::2]), encoding="utf-8")
            second.write_text("".join(json.dumps(row) + "\n" for row in rows[1::2]), encoding="utf-8")
            output = io.StringIO()
            with patch.object(sys, "argv", ["report.py", str(first), str(second), "--replicates", "100"]), redirect_stdout(output):
                self.assertEqual(report_main(), 0)
            report = json.loads(output.getvalue())
        comparison = next(item for item in report["paired_comparisons"] if item["paired_episode_count"] == 2)
        self.assertEqual(comparison["metrics"]["joint_success"]["paired_episodes"], 2)


if __name__ == "__main__":
    unittest.main()
