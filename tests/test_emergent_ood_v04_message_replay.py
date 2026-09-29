from __future__ import annotations

import json
import contextlib
import copy
import hashlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tacit.channel import LocalTCPMessageChannel
from tacit.runtime import ChatCompletion
from tools.cost_report import aggregate
from tools.paired_report import paired_report
from experiments.emergent_ood_v0_4 import replay_usage_messages as replay_module
from experiments.emergent_ood_v0_4.episodes import generate_ledgers, write_ledgers
from experiments.emergent_ood_v0_4.runner import load_protocol_card, select_candidate_sets
from experiments.emergent_ood_v0_4.split import build_split, split_task_id
from experiments.emergent_ood_v0_4.replay_usage_messages import (
    build_compatible_derangement,
    receiver_request,
    run_replay,
)


ATTRIBUTES = ("shape",)


def make_batch():
    values = ["amber", "blue", "coral", "dune"]
    episodes = []
    rows = []
    for index, target in enumerate(values):
        episode_id = f"episode-{index}"
        candidates = [
            {"candidate_id": f"c{index}", "attributes": {"shape": target}},
            {"candidate_id": f"c{(index + 1) % len(values)}", "attributes": {"shape": values[(index + 1) % len(values)]}},
        ]
        episodes.append({
            "sender": {"private_meaning": {"shape": target}},
            "receiver": {"candidates": candidates},
            "gold": {
                "episode_id": episode_id,
                "candidate_id": candidates[0]["candidate_id"],
                "candidate_set_id": f"set-{index}",
                "meaning_id": f"meaning-{index}",
            },
        })
        rows.append({
            "episode_id": episode_id,
            "condition": "usage_only_transfer",
            "protocol_id": "frozen-card-v1",
            "costs": {"message_delivered": True},
            "stratum": {"agent_models": {"sender": "fake-sender", "receiver": "fake-receiver"}},
            "trace": {
                "message": json.dumps({"shape": target}, separators=(",", ":")),
                "target_tuple_for_evaluator": {"shape": target},
                "candidate_ids_in_receiver_order": [candidate["candidate_id"] for candidate in candidates],
            },
        })
    return episodes, rows


class MessageReplayTests(unittest.TestCase):
    def setUp(self):
        self.episodes, self.rows = make_batch()
        self.attributes = ["shape"]
        self.values = {"shape": ["amber", "blue", "coral", "dune"]}
        self.card = {
            "protocol_id": "frozen-card-v1",
            "sender_instruction": "Emit the tuple as JSON.",
            "receiver_instruction": "Decode the JSON tuple.",
        }
        self.examples = [{"meaning": {"shape": "amber"}, "message": '{"shape":"amber"}'}]

    def test_derangement_is_a_hash_seeded_compatible_permutation(self):
        first = build_compatible_derangement(
            self.episodes, self.rows, attributes=self.attributes, seed=9,
            source_results_sha256="a" * 64,
        )
        second = build_compatible_derangement(
            self.episodes, self.rows, attributes=self.attributes, seed=9,
            source_results_sha256="a" * 64,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {row["episode_id"] for row in self.rows})
        self.assertEqual(set(first.values()), {row["episode_id"] for row in self.rows})
        for recipient_id, donor_id in first.items():
            self.assertNotEqual(recipient_id, donor_id)
            episode = next(ep for ep in self.episodes if ep["gold"]["episode_id"] == recipient_id)
            donor = next(row for row in self.rows if row["episode_id"] == donor_id)
            source_meaning = json.loads(donor["trace"]["message"])
            self.assertNotIn(source_meaning, [candidate["attributes"] for candidate in episode["receiver"]["candidates"]])

    def test_rejects_batch_without_a_full_compatible_permutation(self):
        episodes = self.episodes[:2]
        rows = self.rows[:2]
        for episode, row in zip(episodes, rows):
            episode["receiver"]["candidates"] = [
                {"candidate_id": "a", "attributes": {"shape": "amber"}},
                {"candidate_id": "b", "attributes": {"shape": "blue"}},
            ]
            episode["gold"]["candidate_id"] = "a" if episode["sender"]["private_meaning"]["shape"] == "amber" else "b"
            row["trace"]["candidate_ids_in_receiver_order"] = ["a", "b"]
        with self.assertRaisesRegex(ValueError, "compatible|complete"):
            build_compatible_derangement(
                episodes, rows, attributes=self.attributes, seed=1,
                source_results_sha256="b" * 64,
            )

    def test_receiver_prompt_contains_counterfactual_message_and_private_table_only(self):
        prompt = receiver_request(
            message="FOREIGN MESSAGE",
            candidates=self.episodes[0]["receiver"]["candidates"],
            protocol_card=self.card,
            attributes=self.attributes,
            values=self.values,
            usage_examples=self.examples,
        )
        self.assertEqual(len(prompt), 2)
        request = json.loads(prompt[1]["content"])
        self.assertEqual(request["visible_transcript"][0]["message"], "FOREIGN MESSAGE")
        context = json.loads(request["private_context"])
        self.assertEqual(context["candidates"], self.episodes[0]["receiver"]["candidates"])
        self.assertEqual(context["training_examples"], self.examples)
        self.assertNotIn("target_candidate_id", request["private_context"])
        self.assertNotIn("source_episode_id", request["private_context"])

    def test_replay_calls_receiver_once_per_row_and_accounts_reused_message(self):
        class Receiver:
            model = "fake-receiver"

            def __init__(self):
                self.prompts = []

            def complete(self, messages):
                self.prompts.append(messages)
                request = json.loads(messages[-1]["content"])
                private = json.loads(request["private_context"])
                candidate = private["candidates"][1]
                return ChatCompletion(candidate["candidate_id"], self.model, 90, 2, 0.01)

        receiver = Receiver()
        rows, assignment = run_replay(
            episodes=self.episodes,
            source_rows=self.rows,
            receiver_model=receiver,
            attributes=self.attributes,
            values=self.values,
            protocol_card=self.card,
            usage_examples=self.examples,
            receiver_tokenizer_id="fake-tokenizer",
            model_population_id="fake-population",
            stage="test",
            split_seed=1,
            task_seed=2,
            task_id="test-task",
            ontology_id=None,
            target_support_size=4,
            wire_budget_bytes=4096,
            seed=3,
            source_results_sha256="c" * 64,
        )
        self.assertEqual(len(receiver.prompts), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["costs"]["model_call_count"] for row in rows), 4)
        self.assertTrue(all(row["costs"]["sender_generation_reused_from_source_run"] for row in rows))
        self.assertTrue(all(row["condition"] == "usage_only_transfer_message_deranged" for row in rows))
        self.assertTrue(all(row["trace"]["source_episode_id"] == assignment[row["episode_id"]] for row in rows))
        self.assertTrue(all(row["outcome"]["exact_selection"] is False for row in rows))
        report = aggregate(rows)
        group = report["groups"][0]
        self.assertEqual(group["inference"]["model_calls"], 4)
        self.assertGreater(group["channel"]["wire_bytes"]["observed_sum"], 0)

        source_by_id = {row["episode_id"]: row for row in self.rows}
        original_rows = []
        channel = LocalTCPMessageChannel(lambda _: None)
        for replay_row in rows:
            original = copy.deepcopy(replay_row)
            source = source_by_id[original["episode_id"]]
            original["condition"] = "usage_only_transfer"
            original["protocol"]["policy_id"] = "usage_only_transfer"
            original["outcome"].update({
                "answer_format_valid": True,
                "answer_candidate_id": original["outcome"]["target_candidate_id"],
                "exact_selection": True,
                "joint_success": True,
                "answer_score": 1.0,
            })
            original_message = source["trace"]["message"]
            original_transmission = channel.measure(
                original_message, protocol_id=source["protocol_id"], round_number=1,
                sender="sender", recipient="receiver",
            )
            original["transmissions"] = [original_transmission.cost_record(
                recipient_tokenizer="fake-tokenizer",
            )]
            sender_call = {
                "agent": "sender", "round": 1, "stage": "dialogue_turn",
                "model": "fake-sender", "tokenizer": "fake-sender-tokenizer",
                "input_tokens": 50, "output_tokens": 5, "service_seconds": 0.01,
                "retry": False, "truncated": False,
            }
            receiver_call = dict(original["model_calls"][0], stage="final_answer")
            original["model_calls"] = [sender_call, receiver_call]
            original["stratum"]["agent_models"] = {
                "sender": "fake-sender", "receiver": "fake-receiver",
            }
            original_rows.append(original)
        paired = paired_report([*original_rows, *rows], replicates=100, seed=7)
        comparison = paired["comparisons"][0]
        self.assertEqual(comparison["paired_episode_count"], 4)
        self.assertTrue(comparison["control_alignment"]["model_strata_matched"])
        self.assertAlmostEqual(
            comparison["metrics"]["wire_bytes"]["mean_left_minus_right"], 0.0,
        )

    def test_completed_receiver_rows_are_not_repeated_after_interruption(self):
        class InterruptOnceReceiver:
            model = "fake-receiver"

            def __init__(self):
                self.attempts = 0
                self.interrupted = False

            def complete(self, messages):
                self.attempts += 1
                if self.attempts == 2 and not self.interrupted:
                    self.interrupted = True
                    raise RuntimeError("synthetic interruption")
                request = json.loads(messages[-1]["content"])
                private = json.loads(request["private_context"])
                candidate = private["candidates"][1]
                return ChatCompletion(candidate["candidate_id"], self.model, 50, 2, 0.01)

        receiver = InterruptOnceReceiver()
        completed = []
        arguments = {
            "episodes": self.episodes,
            "source_rows": self.rows,
            "receiver_model": receiver,
            "attributes": self.attributes,
            "values": self.values,
            "protocol_card": self.card,
            "usage_examples": self.examples,
            "receiver_tokenizer_id": "fake-tokenizer",
            "model_population_id": "fake-population",
            "stage": "test",
            "split_seed": 1,
            "task_seed": 2,
            "task_id": "test-task",
            "ontology_id": None,
            "target_support_size": 4,
            "wire_budget_bytes": 4096,
            "seed": 3,
            "source_results_sha256": "d" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            replay_module.run_replay(
                **arguments, on_row_complete=lambda row: completed.append(row),
            )
        self.assertEqual(len(completed), 1)
        rows, _ = replay_module.run_replay(
            **arguments,
            completed_rows=completed,
            on_row_complete=lambda row: completed.append(row),
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(completed), 4)
        self.assertEqual(receiver.attempts, 5)

    def test_checkpoint_is_atomic_hash_bound_and_rejects_a_different_plan(self):
        project_cache = Path(__file__).resolve().parents[1] / ".cache"
        project_cache.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=project_cache) as temporary:
            checkpoint_path = Path(temporary) / "replay.checkpoint.json"
            run_config = {"episode_ids": ["a", "b"], "seed": 17}
            checkpoint = {
                "schema": replay_module.CHECKPOINT_SCHEMA,
                "run_signature": replay_module._signature(run_config),
                "run_config": run_config,
                "rows": [],
                "resource_preflight_sha256s": ["e" * 64],
                "resumed": False,
            }
            replay_module._write_checkpoint(checkpoint_path, checkpoint)
            recovered = replay_module._read_checkpoint(
                checkpoint_path, run_signature=checkpoint["run_signature"],
            )
            self.assertEqual(recovered["run_config"], run_config)
            self.assertEqual(recovered["rows"], [])
            with self.assertRaisesRegex(ValueError, "integrity"):
                replay_module._read_checkpoint(checkpoint_path, run_signature="f" * 64)
            damaged = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            damaged["resource_preflight_sha256s"].append("bad")
            checkpoint_path.write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                replay_module._read_checkpoint(
                    checkpoint_path, run_signature=checkpoint["run_signature"],
                )

    def test_cli_offline_plan_joins_adjacent_frozen_single_set_batches(self):
        project_cache = Path(__file__).resolve().parents[1] / ".cache"
        project_cache.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=project_cache) as temporary:
            root = Path(temporary)
            split = build_split(seed=18)
            bundle = generate_ledgers(
                split=split, task_key=bytes(range(32)), task_seed=9,
                k=4, sets_per_stage=3,
            )
            input_dir = root / "evaluation"
            write_ledgers(bundle, input_dir)
            input_manifest_sha256 = hashlib.sha256((input_dir / "manifest.json").read_bytes()).hexdigest()
            card_path = root / "card.json"
            card_path.write_text(json.dumps({
                "schema": "tlu.shared_protocol_card.v1",
                "protocol_id": "cli-plan-card-v1",
                "sender_instruction": "Encode every value.",
                "receiver_instruction": "Decode every value.",
            }), encoding="utf-8")
            card, card_digest = load_protocol_card(card_path)
            train_meaning = bundle["sender"]["train"][0]["private_meaning"]
            train_tuple = tuple(train_meaning[axis] for axis in split["attributes"])
            meaning_id = next(
                row["meaning_id"] for row in split["meanings"]
                if tuple(row["values"]) == train_tuple
            )
            usage_path = root / "usage.json"
            usage_path.write_text(json.dumps({
                "schema": "tlu.usage_examples.v1",
                "protocol_id": card["protocol_id"],
                "protocol_card_sha256": card_digest,
                "training_split_sha256": split["split_sha256"],
                "training_episode_manifest_sha256": input_manifest_sha256,
                "acquisition": {
                    "method": "programmatic", "model_id": "", "tokenizer_id": "",
                    "generation_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "service_seconds": 0.0, "wall_seconds": 0.0,
                },
                "examples": [{
                    "meaning_id": meaning_id, "meaning": train_meaning,
                    "message": json.dumps(train_meaning, separators=(",", ":")),
                }],
            }), encoding="utf-8")
            episodes = select_candidate_sets(bundle, "test", 3)
            source_paths = []
            for set_offset in range(3):
                selected = select_candidate_sets(bundle, "test", 1, set_offset=set_offset)
                rows = []
                for episode in selected:
                    gold = episode["gold"]
                    meaning = episode["sender"]["private_meaning"]
                    candidate_ids = [item["candidate_id"] for item in episode["receiver"]["candidates"]]
                    message = json.dumps(meaning, separators=(",", ":"))
                    transmission = LocalTCPMessageChannel(lambda _: None).measure(
                        message, protocol_id=card["protocol_id"], round_number=1,
                        sender="sender", recipient="receiver",
                    ).cost_record(recipient_tokenizer="fake-tokenizer")
                    rows.append({
                        "schema_version": "tlu.costs.v3",
                        "schema": "tlu.emergent-ood-run.v0.4",
                        "experiment_id": replay_module.EXPERIMENT_ID,
                        "episode_id": gold["episode_id"],
                        "condition": "usage_only_transfer",
                        "stage": "test",
                        "split_seed": 18,
                        "task_seed": 9,
                        "candidate_set_id": gold["candidate_set_id"],
                        "meaning_id": gold["meaning_id"],
                        "protocol_id": card["protocol_id"],
                        "outcome": {"target_candidate_id": gold["candidate_id"]},
                        "costs": {
                            "message_delivered": True,
                            "communication_budget_bytes": 4096,
                            "generated_message_bytes": len(message.encode("utf-8")),
                            "delivered_payload_bytes": len(message.encode("utf-8")),
                            "application_wire_bytes": transmission["payload_bytes"] + transmission["framing_bytes"],
                        },
                        "trace": {
                            "message": message,
                            "target_tuple_for_evaluator": meaning,
                            "candidate_ids_in_receiver_order": candidate_ids,
                            "transmissions": [transmission],
                        },
                        "transmissions": [transmission],
                        "stratum": {"model_population_id": "test-population", "task_id": split_task_id(split)},
                        "model_calls": [
                            {"agent": "sender", "stage": "dialogue_turn", "model": "fake-sender", "tokenizer": "fake-sender-tokenizer", "truncated": False},
                            {"agent": "receiver", "stage": "final_answer", "model": "fake-receiver", "tokenizer": "fake-tokenizer", "truncated": False},
                        ],
                    })
                source_path = root / f"source-{set_offset}.jsonl"
                source_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
                source_path.write_bytes(source_bytes)
                source_manifest = {
                    "schema": "tlu.emergent-ood-run-manifest.v0.4",
                    "results_file": source_path.name,
                    "results_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "result_records": len(rows),
                    "experiment_id": replay_module.EXPERIMENT_ID,
                    "stage": "test",
                    "conditions": ["usage_only_transfer"],
                    "split_seed": 18,
                    "input_episode_manifest_sha256": input_manifest_sha256,
                    "protocol_card_sha256": card_digest,
                    "usage_examples_sha256": hashlib.sha256(usage_path.read_bytes()).hexdigest(),
                    "usage_reuse_horizon": 12,
                    "split_sha256": split["split_sha256"],
                    "ontology_id": split.get("ontology_id"),
                    "task_seed": 9,
                    "candidate_count": 4,
                    "task_id": split_task_id(split),
                    "candidate_sets": 1,
                    "candidate_set_offset": set_offset,
                    "communication_budget_bytes": 4096,
                    "sender_model": "fake-sender",
                    "receiver_model": "fake-receiver",
                    "sender_tokenizer_id": "fake-sender-tokenizer",
                    "receiver_tokenizer_id": "fake-tokenizer",
                    "model_population_id": "test-population",
                    "temperature": 0.0,
                    "sender_max_tokens": 160,
                    "receiver_max_tokens": 48,
                }
                source_path.with_suffix(".jsonl.manifest.json").write_text(
                    json.dumps(source_manifest), encoding="utf-8",
                )
                source_paths.append(source_path)

            output_path = root / "planned-replay.jsonl"
            argv = [
                "replay", "--input-dir", str(input_dir), "--split-seed", "18",
                "--stage", "test", "--sets", "3", "--set-offset", "0",
                "--source-results", *(str(path) for path in source_paths),
                "--protocol-card", str(card_path), "--usage-examples", str(usage_path),
                "--usage-reuse-horizon", "12", "--seed", "31", "--output", str(output_path),
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(replay_module.main(), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["mode"], "dry-run")
            self.assertEqual(report["receiver_calls_planned"], 12)
            self.assertTrue(report["compatible_derangement_complete"])
            self.assertFalse(report["inference_started"])
            self.assertEqual(len(episodes), len(report["recipient_to_donor_episode_ids"]))


if __name__ == "__main__":
    unittest.main()
