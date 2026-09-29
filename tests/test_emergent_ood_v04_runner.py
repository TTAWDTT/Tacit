from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.emergent_ood_v0_4.episodes import generate_ledgers, verify_ledgers, write_ledgers
import experiments.emergent_ood_v0_4.runner as runner_module
from experiments.emergent_ood_v0_4.runner import (
    _strict_json_tuple,
    conflict_graph_summary,
    load_protocol_card,
    load_episode_bundle,
    run_condition,
    select_candidate_sets,
    validate_capability_ledger,
)
from experiments.emergent_ood_v0_4.split import build_split
from tacit.runtime import ChatCompletion
from tools.cost_report import aggregate
from tools.paired_report import paired_report


class FakeSender:
    model = "fake-sender"

    def __init__(self, condition, attributes, values):
        self.condition = condition
        self.attributes = attributes
        self.values = values
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        request = json.loads(messages[-1]["content"])
        private = json.loads(request["private_context"])
        target = private["private_meaning"]
        if self.condition == "natural_language":
            text = ", ".join(f"{attribute} {target[attribute]}" for attribute in self.attributes)
        elif self.condition == "json":
            ordered = {attribute: target[attribute] for attribute in self.attributes}
            text = json.dumps(ordered, separators=(",", ":"))
        else:
            text = "".join(str(list(self.values[axis]).index(target[axis])) for axis in self.attributes)
        return ChatCompletion(text, self.model, 21, len(text), 0.01)


class FakeReceiver:
    model = "fake-receiver"

    def __init__(self, condition, attributes, values):
        self.condition = condition
        self.attributes = attributes
        self.values = values
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        request = json.loads(messages[-1]["content"])
        private = json.loads(request["private_context"])
        candidates = private["candidates"]
        if "calibration_target" in private:
            meaning = private["calibration_target"]
        elif self.condition == "no_message":
            return ChatCompletion(candidates[0]["candidate_id"], self.model, 30, 1, 0.01)
        else:
            message = request["visible_transcript"][0]["message"]
            if self.condition == "natural_language":
                meaning = {}
                for axis in self.attributes:
                    meaning[axis] = next(value for value in self.values[axis] if value in message)
            elif self.condition == "json":
                meaning = json.loads(message)
            else:
                meaning = {axis: self.values[axis][int(code)] for axis, code in zip(self.attributes, message)}
        candidate = next(row for row in candidates if row["attributes"] == meaning)
        return ChatCompletion(candidate["candidate_id"], self.model, 32, 6, 0.01)


class MissingUsageReceiver(FakeReceiver):
    def complete(self, messages):
        completion = super().complete(messages)
        return ChatCompletion(completion.text, completion.model, None, None, None)


class EmergentOODV04RunnerTests(unittest.TestCase):
    def setUp(self):
        self.split = build_split(seed=17)
        self.bundle = generate_ledgers(split=self.split, task_key=bytes(range(32)), task_seed=9, k=4, sets_per_stage=3)
        verify_ledgers(self.bundle)
        self.episode = select_candidate_sets(self.bundle, "validation", 1)[0]
        self.attributes = self.split["attributes"]
        self.values = self.split["values_by_attribute"]

    def _run(self, condition, episode=None, stage="validation"):
        episode = self.episode if episode is None else episode
        mock_condition = "json" if condition == "shared_protocol_card" else condition
        sender = None if condition in {"full_information", "no_message"} else FakeSender(mock_condition, self.attributes, self.values)
        receiver = FakeReceiver(mock_condition, self.attributes, self.values)
        card = None
        if condition == "shared_protocol_card":
            card = {
                "protocol_id": "frozen-card-test-v1",
                "sender_instruction": "Encode the tuple as compact JSON.",
                "receiver_instruction": "Decode the tuple from the message.",
            }
        row = run_condition(
            episode=episode,
            condition=condition,
            stage=stage,
            sender_model=sender,
            receiver_model=receiver,
            sender_tokenizer_id="fake-sender-tokenizer-v1" if sender is not None else None,
            receiver_tokenizer_id="fake-receiver-tokenizer-v1",
            protocol_card=card,
            attributes=self.attributes,
            values=self.values,
            split_seed=17,
            task_seed=9,
            model_population_id="fake-test-population",
        )
        return row, sender, receiver

    def test_message_conditions_route_exact_payload_and_score_end_to_end(self):
        for condition in ("natural_language", "json", "symbolic"):
            with self.subTest(condition=condition):
                row, sender, receiver = self._run(condition)
                self.assertTrue(row["outcome"]["answer_format_valid"])
                self.assertTrue(row["outcome"]["exact_selection"])
                self.assertTrue(row["costs"]["message_delivered"])
                self.assertGreater(row["costs"]["application_wire_bytes"], row["costs"]["delivered_payload_bytes"])
                self.assertEqual(row["costs"]["model_call_count"], 2)
                self.assertEqual(row["costs"]["complete_input_tokens"], 53)
                cost_report = aggregate([row])
                self.assertEqual(cost_report["input_schema_version"], "tlu.costs.v3")
                self.assertEqual(cost_report["groups"][0]["episodes"], 1)
                self.assertEqual(
                    [call["tokenizer"] for call in row["costs"]["model_calls"]],
                    ["fake-sender-tokenizer-v1", "fake-receiver-tokenizer-v1"],
                )
                self.assertEqual(len(sender.prompts), 1)
                self.assertEqual(len(receiver.prompts), 1)

    def test_message_conditions_keep_private_roles_and_evaluator_labels_separate(self):
        row, sender, receiver = self._run("json")
        sender_view = json.loads(sender.prompts[0][-1]["content"])["private_context"]
        receiver_view = json.loads(receiver.prompts[0][-1]["content"])["private_context"]
        self.assertIn("private_meaning", sender_view)
        self.assertNotIn("candidates", sender_view)
        self.assertIn("candidates", receiver_view)
        self.assertNotIn("private_meaning", receiver_view)
        for prompt in (sender.prompts[0], receiver.prompts[0]):
            text = json.dumps(prompt)
            for evaluator_key in ("episode_id", "meaning_id", "candidate_set_id", "target_candidate_id", "gold", "stage"):
                self.assertNotIn(evaluator_key, text)
        self.assertIn("candidate_set_id", row)

    def test_controls_use_one_receiver_call_and_no_transmission(self):
        full, _, full_receiver = self._run("full_information")
        no_message, _, no_receiver = self._run("no_message")
        self.assertTrue(full["outcome"]["exact_selection"])
        self.assertEqual(full["costs"]["model_call_count"], 1)
        self.assertEqual(full["costs"]["application_wire_bytes"], 0)
        self.assertEqual(len(full_receiver.prompts), 1)
        self.assertEqual(json.loads(full_receiver.prompts[0][-1]["content"])["visible_transcript"], [])
        self.assertEqual(no_message["costs"]["model_call_count"], 1)
        self.assertEqual(no_message["costs"]["application_wire_bytes"], 0)
        self.assertEqual(len(no_receiver.prompts), 1)

    def test_missing_provider_usage_is_unknown_not_silently_underreported(self):
        sender = FakeSender("json", self.attributes, self.values)
        receiver = MissingUsageReceiver("json", self.attributes, self.values)
        row = run_condition(
            episode=self.episode,
            condition="json",
            stage="validation",
            sender_model=sender,
            receiver_model=receiver,
            sender_tokenizer_id="fake-sender-tokenizer-v1",
            receiver_tokenizer_id="fake-receiver-tokenizer-v1",
            attributes=self.attributes,
            values=self.values,
            split_seed=17,
            task_seed=9,
            model_population_id="fake-test-population",
        )
        self.assertIsNone(row["costs"]["complete_input_tokens"])
        self.assertIsNone(row["costs"]["complete_output_tokens"])
        self.assertEqual(row["costs"]["calls_with_input_token_usage"], 1)
        self.assertEqual(row["costs"]["calls_with_output_token_usage"], 1)

    def test_costs_v3_paired_analysis_keeps_split_as_the_uncertainty_unit(self):
        episodes = select_candidate_sets(self.bundle, "validation", 1)
        records = []
        for episode in episodes:
            for condition in ("no_message", "natural_language"):
                row, _, _ = self._run(condition, episode)
                records.append(row)
        report = paired_report(records, replicates=100, seed=5)
        self.assertEqual(report["input_schema_version"], "tlu.costs.v3")
        self.assertEqual(len(report["comparisons"]), 1)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 4)
        self.assertEqual(comparison["paired_inference_cluster_count"], 1)
        self.assertIsNone(comparison["metrics"]["joint_success"]["ci95_low"])

    def _write_capability_fixture(self, *, corrupt=False):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=cache_root)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "capability.jsonl"
        calibration = select_candidate_sets(self.bundle, "train", 3)
        records = [self._run("full_information", episode, stage="train")[0] for episode in calibration]
        if corrupt:
            records[0] = {**records[0], "outcome": {**records[0]["outcome"], "joint_success": False}}
        input_manifest_sha256 = "a" * 64
        runner_module._write_results(
            records,
            output,
            force=False,
            manifest={
                "experiment_id": runner_module.EXPERIMENT_ID,
                "stage": "train",
                "conditions": ["full_information"],
                "candidate_sets": 3,
                "candidate_count": 4,
                "split_seed": 17,
                "split_sha256": self.split["split_sha256"],
                "task_seed": 9,
                "task_key_id": self.bundle["manifest"]["task_key_id"],
                "input_episode_manifest_sha256": input_manifest_sha256,
                "receiver_model": "fake-receiver",
                "receiver_tokenizer_id": "fake-receiver-tokenizer-v1",
                "model_population_id": "fake-test-population",
            },
        )
        return output, calibration, input_manifest_sha256

    def test_train_only_capability_ledger_is_verified_and_disjoint(self):
        output, calibration, input_manifest_sha256 = self._write_capability_fixture()
        validation = select_candidate_sets(self.bundle, "validation", 1)
        validate_capability_ledger(
            output,
            expected_calibration_episodes=calibration,
            evaluation_episodes=validation,
            input_manifest_sha256=input_manifest_sha256,
            split_seed=17,
            split_sha256=self.split["split_sha256"],
            task_seed=9,
            task_key_id=self.bundle["manifest"]["task_key_id"],
            receiver_model="fake-receiver",
            receiver_tokenizer_id="fake-receiver-tokenizer-v1",
            model_population_id="fake-test-population",
        )

    def test_capability_ledger_rejects_failures_overlap_and_tampered_hash(self):
        validation = select_candidate_sets(self.bundle, "validation", 1)
        output, calibration, input_manifest_sha256 = self._write_capability_fixture(corrupt=True)
        with self.assertRaisesRegex(ValueError, "perfect"):
            validate_capability_ledger(
                output,
                expected_calibration_episodes=calibration,
                evaluation_episodes=validation,
                input_manifest_sha256=input_manifest_sha256,
                split_seed=17,
                split_sha256=self.split["split_sha256"],
                task_seed=9,
                task_key_id=self.bundle["manifest"]["task_key_id"],
                receiver_model="fake-receiver",
                receiver_tokenizer_id="fake-receiver-tokenizer-v1",
                model_population_id="fake-test-population",
            )
        output, calibration, input_manifest_sha256 = self._write_capability_fixture()
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_capability_ledger(
                output,
                expected_calibration_episodes=calibration,
                evaluation_episodes=calibration,
                input_manifest_sha256=input_manifest_sha256,
                split_seed=17,
                split_sha256=self.split["split_sha256"],
                task_seed=9,
                task_key_id=self.bundle["manifest"]["task_key_id"],
                receiver_model="fake-receiver",
                receiver_tokenizer_id="fake-receiver-tokenizer-v1",
                model_population_id="fake-test-population",
            )
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["results_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_capability_ledger(
                output,
                expected_calibration_episodes=calibration,
                evaluation_episodes=validation,
                input_manifest_sha256=input_manifest_sha256,
                split_seed=17,
                split_sha256=self.split["split_sha256"],
                task_seed=9,
                task_key_id=self.bundle["manifest"]["task_key_id"],
                receiver_model="fake-receiver",
                receiver_tokenizer_id="fake-receiver-tokenizer-v1",
                model_population_id="fake-test-population",
            )

    def test_missing_capability_gate_rejects_before_endpoint_client_creation(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            input_dir = Path(temporary) / "episodes"
            write_ledgers(self.bundle, input_dir)
            args = [
                "runner",
                "--input-dir", str(input_dir),
                "--split-seed", "17",
                "--stage", "validation",
                "--conditions", "natural_language",
                "--execute",
                "--receiver-model", "fake-receiver",
                "--receiver-tokenizer-id", "fake-receiver-tokenizer-v1",
                "--sender-model", "fake-sender",
                "--sender-tokenizer-id", "fake-sender-tokenizer-v1",
            ]
            with patch("sys.argv", args), patch.object(
                runner_module, "OpenAICompatibleClient"
            ) as endpoint_client, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner_module.main()
            endpoint_client.assert_not_called()

    def test_incomplete_candidate_set_selection_is_rejected(self):
        corrupt = {**self.bundle, "gold": {**self.bundle["gold"]}}
        corrupt["gold"]["validation"] = corrupt["gold"]["validation"][:-1]
        with self.assertRaises(ValueError):
            select_candidate_sets(corrupt, "validation", 1)

    def test_single_candidate_set_has_exact_four_color_zero_error_floor(self):
        episodes = select_candidate_sets(self.bundle, "validation", 1)
        graph = conflict_graph_summary(episodes)
        self.assertEqual(graph["observed_target_vertices"], 4)
        self.assertEqual(graph["cooccurring_target_pairs"], 6)
        self.assertEqual(graph["chromatic_number"], 4)
        self.assertTrue(graph["chromatic_number_exact"])
        self.assertTrue(graph["complete_pair_graph_on_observed_vertices"])
        self.assertEqual(graph["fixed_width_zero_error_payload_floor_bits"], 2)

    def test_coloring_solver_uses_matching_certified_bounds_without_search(self):
        episodes = select_candidate_sets(self.bundle, "validation", 1)
        original_limit = runner_module.MAX_EXACT_COLORING_VERTICES
        try:
            runner_module.MAX_EXACT_COLORING_VERTICES = 2
            graph = conflict_graph_summary(episodes)
        finally:
            runner_module.MAX_EXACT_COLORING_VERTICES = original_limit
        self.assertEqual(graph["chromatic_number"], 4)
        self.assertTrue(graph["chromatic_number_exact"])
        self.assertEqual(graph["minimum_clique_lower_bound"], 4)
        self.assertEqual(graph["greedy_coloring_upper_bound"], 4)

    def test_bundle_loader_checks_hashes_and_meaning_partitions(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            output = Path(temporary) / "episodes"
            write_ledgers(self.bundle, output)
            loaded, loaded_split = load_episode_bundle(output, split_seed=17)
            self.assertEqual(loaded_split["split_sha256"], self.split["split_sha256"])
            self.assertEqual(len(loaded["gold"]["test"]), 12)
            with self.assertRaises(FileExistsError):
                write_ledgers(self.bundle, output)
            receiver_test = output / "receiver_test.jsonl"
            receiver_test.write_text(receiver_test.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_episode_bundle(output, split_seed=17)

    def test_json_sender_fidelity_rejects_duplicate_keys_but_accepts_unordered_fields(self):
        target = {axis: self.values[axis][0] for axis in self.attributes}
        unordered = json.dumps(dict(reversed(list(target.items()))), separators=(",", ":"))
        semantic_valid, canonical = _strict_json_tuple(unordered, self.attributes, target)
        self.assertTrue(semantic_valid)
        self.assertFalse(canonical)
        duplicate = '{"shape":"circle","shape":"square"}'
        semantic_valid, canonical = _strict_json_tuple(duplicate, self.attributes, target)
        self.assertFalse(semantic_valid)
        self.assertFalse(canonical)

    def test_frozen_protocol_card_is_hashed_and_can_drive_the_shared_card_arm(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
            card_path = Path(temporary) / "card.json"
            card_path.write_text(json.dumps({
                "schema": "tlu.shared_protocol_card.v1",
                "protocol_id": "frozen-card-test-v1",
                "sender_instruction": "Encode the tuple as compact JSON.",
                "receiver_instruction": "Decode the tuple from the message.",
            }), encoding="utf-8")
            card, digest = load_protocol_card(card_path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(card["protocol_id"], "frozen-card-test-v1")
        row, _, _ = self._run("shared_protocol_card")
        self.assertEqual(row["protocol_id"], "frozen-card-test-v1")
        self.assertTrue(row["outcome"]["exact_selection"])


if __name__ == "__main__":
    unittest.main()
