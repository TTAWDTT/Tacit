from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.emergent_ood_v0_4 import episodes
from experiments.emergent_ood_v0_4.induce_protocol_cards import run_induction
from experiments.emergent_ood_v0_4.nl_feedback import FEEDBACK_SCHEMA, build_feedback
from experiments.emergent_ood_v0_4.select_nl_search_parent import freeze_training_frontier
from experiments.emergent_ood_v0_4.runner import EXPERIMENT_ID, SCORER_ID
from experiments.emergent_ood_v0_4.split import build_split


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    return payload


class NlFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.directory = Path(self.temp.name)
        self.key = bytes(range(32))
        self.split_seed = 71
        split = build_split(seed=self.split_seed)
        self.bundle_dir = self.directory / "episodes"
        bundle = episodes.generate_ledgers(split=split, task_key=self.key, task_seed=9, k=4, sets_per_stage=2)
        self.bundle_manifest = episodes.write_ledgers(bundle, self.bundle_dir)
        self.card = {
            "schema": "tlu.shared_protocol_card.v1",
            "protocol_id": "nl-test-card",
            "sender_instruction": "Describe each field in a complete English sentence.",
            "receiver_instruction": "Match all fields and return the corresponding candidate_id.",
        }
        self.card_path = self.directory / "card.json"
        self.card_bytes = (json.dumps(self.card, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.card_path.write_bytes(self.card_bytes)
        self.card_sha = hashlib.sha256(self.card_bytes).hexdigest()
        self.induction_manifest = {
            "schema": "tlu.emergent-ood-induced-card-set.v1",
            "split_seed": self.split_seed,
            "split_sha256": split["split_sha256"],
            "protocol_family": "plain_english",
            "optimization_round": 1,
            "max_optimization_rounds": 2,
            "candidate_count": 4,
            "feedback_sha256": None,
            "candidate_cards": [{"protocol_id": self.card["protocol_id"], "sha256": self.card_sha}],
        }
        self.induction_path = self.directory / "induction-manifest.json"
        self.induction_path.write_text(json.dumps(self.induction_manifest), encoding="utf-8")

        train = bundle
        selected_ids = {train["gold"]["train"][i]["candidate_set_id"] for i in range(4)}
        output_rows = []
        for index, (sender, receiver, gold) in enumerate(zip(
            train["sender"]["train"], train["receiver"]["train"], train["gold"]["train"]
        )):
            if gold["candidate_set_id"] not in selected_ids:
                continue
            # Retain one error so the feedback path has a meaningful signal.
            predicted_id = gold["candidate_id"] if len(output_rows) else next(
                c["candidate_id"] for c in receiver["candidates"] if c["candidate_id"] != gold["candidate_id"]
            )
            output_rows.append({
                "schema": "tlu.emergent-ood-run.v0.4",
                "experiment_id": EXPERIMENT_ID,
                "stage": "train",
                "split_seed": self.split_seed,
                "task_seed": 9,
                "condition": "shared_protocol_card",
                "protocol_id": self.card["protocol_id"],
                "episode_id": gold["episode_id"],
                "candidate_set_id": gold["candidate_set_id"],
                "meaning_id": gold["meaning_id"],
                "outcome": {
                    "answer_format_valid": True,
                    "answer_candidate_id": predicted_id,
                    "target_candidate_id": gold["candidate_id"],
                    "exact_selection": predicted_id == gold["candidate_id"],
                    "joint_success": predicted_id == gold["candidate_id"],
                    "answer_score": float(predicted_id == gold["candidate_id"]),
                },
                "trace": {
                    "message": "The shade is blue and the pattern is striped.",
                    "target_tuple_for_evaluator": sender["private_meaning"],
                    "candidate_ids_in_receiver_order": [c["candidate_id"] for c in receiver["candidates"]],
                },
                "stratum": {
                    "scorer_id": SCORER_ID,
                    "task_id": "four-attribute-higher-order-meaning-matching-v1",
                    "model_population_id": "test-population",
                    "agent_models": {"sender": "test-model", "receiver": "test-model"},
                },
                "protocol": {"decoder_id": SCORER_ID},
                "costs": {
                    "model_call_count": 2,
                    "application_wire_bytes": 100,
                    "complete_input_tokens": None,
                    "complete_output_tokens": None,
                    "complete_reported_service_seconds": None,
                },
                "runtime": {"wall_seconds": 0.1},
            })
        self.run_path = self.directory / "run.jsonl"
        run_bytes = _write_jsonl(self.run_path, output_rows)
        manifest_path = self.run_path.with_suffix(".jsonl.manifest.json")
        run_manifest = {
            "schema": "tlu.emergent-ood-run-manifest.v0.4",
            "experiment_id": EXPERIMENT_ID,
            "results_file": self.run_path.name,
            "results_sha256": hashlib.sha256(run_bytes).hexdigest(),
            "result_records": len(output_rows),
            "stage": "train",
            "conditions": ["shared_protocol_card"],
            "split_seed": self.split_seed,
            "split_sha256": split["split_sha256"],
            "input_episode_manifest_sha256": hashlib.sha256(
                (self.bundle_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "protocol_card_sha256": self.card_sha,
            "task_key_id": hashlib.sha256(self.key).hexdigest()[:16],
            "task_seed": 9,
            "candidate_count": 4,
            "candidate_set_offset": 0,
            "candidate_sets": 1,
            "model_population_id": "test-population",
            "sender_model": "test-model",
            "receiver_model": "test-model",
            "sender_tokenizer_id": "test-tokenizer",
            "receiver_tokenizer_id": "test-tokenizer",
            "communication_budget_bytes": 4096,
        }
        manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_feedback_reads_training_ledgers_only_and_round_trips_to_next_induction(self) -> None:
        output = self.directory / "feedback.json"
        original_read_bytes = Path.read_bytes

        def guarded_read(path: Path) -> bytes:
            if path.name.endswith("_validation.jsonl") or path.name.endswith("_test.jsonl"):
                raise AssertionError(f"non-training role file was opened: {path.name}")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", guarded_read):
            feedback = build_feedback(
                episode_dir=self.bundle_dir,
                run_path=self.run_path,
                card_path=self.card_path,
                induction_manifest_path=self.induction_path,
                split_seed=self.split_seed,
            )
            output.write_text(json.dumps(feedback), encoding="utf-8")
            key_path = self.directory / "task.key"
            key_path.write_bytes(self.key)
            plan = run_induction(
                split_seed=self.split_seed,
                task_key_path=key_path,
                example_count=2,
                candidate_count=4,
                output_dir=self.directory / "dry-run-output",
                execute=False,
                protocol_family="plain_english",
                feedback_path=output,
                episode_dir=self.bundle_dir,
            )

        self.assertEqual(feedback["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(feedback["split"], "train")
        self.assertFalse(feedback["validation_files_opened"])
        self.assertFalse(feedback["test_files_opened"])
        self.assertEqual(feedback["episodes"], 4)
        self.assertEqual(feedback["failure_count"], 1)
        self.assertEqual(feedback["failure_examples"][0]["failure_type"], "wrong_candidate")
        self.assertEqual(plan["optimization_round"], 2)
        self.assertEqual(plan["feedback_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertFalse(plan["inference_started"])
        self.assertEqual(plan["test_examples_in_prompt"], 0)

    def test_paired_train_frontier_retains_quality_cost_tradeoff(self) -> None:
        first_feedback_path = self.directory / "feedback-first.json"
        second_card = {**self.card, "protocol_id": "nl-test-card-b"}
        second_card_path = self.directory / "card-b.json"
        second_card_bytes = (json.dumps(second_card, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        second_card_path.write_bytes(second_card_bytes)
        induction = json.loads(self.induction_path.read_text(encoding="utf-8"))
        induction["candidate_cards"].append({
            "protocol_id": second_card["protocol_id"],
            "sha256": hashlib.sha256(second_card_bytes).hexdigest(),
        })
        self.induction_path.write_text(json.dumps(induction), encoding="utf-8")

        first = build_feedback(
            episode_dir=self.bundle_dir,
            run_path=self.run_path,
            card_path=self.card_path,
            induction_manifest_path=self.induction_path,
            split_seed=self.split_seed,
        )
        first_feedback_path.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")

        second_rows = [json.loads(line) for line in self.run_path.read_text(encoding="utf-8").splitlines()]
        for row in second_rows:
            row["protocol_id"] = second_card["protocol_id"]
            row["outcome"]["answer_candidate_id"] = row["outcome"]["target_candidate_id"]
            row["outcome"]["exact_selection"] = True
            row["outcome"]["joint_success"] = True
            row["outcome"]["answer_score"] = 1.0
            row["costs"]["application_wire_bytes"] = 90
        second_run_path = self.directory / "run-b.jsonl"
        second_run_bytes = _write_jsonl(second_run_path, second_rows)
        second_manifest_path = second_run_path.with_suffix(".jsonl.manifest.json")
        second_manifest = json.loads(self.run_path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
        second_manifest.update({
            "results_file": second_run_path.name,
            "results_sha256": hashlib.sha256(second_run_bytes).hexdigest(),
            "result_records": len(second_rows),
            "protocol_card_sha256": hashlib.sha256(second_card_bytes).hexdigest(),
        })
        second_manifest_path.write_text(json.dumps(second_manifest), encoding="utf-8")
        second = build_feedback(
            episode_dir=self.bundle_dir,
            run_path=second_run_path,
            card_path=second_card_path,
            induction_manifest_path=self.induction_path,
            split_seed=self.split_seed,
        )
        second_feedback_path = self.directory / "feedback-second.json"
        second_feedback_path.write_text(json.dumps(second, sort_keys=True), encoding="utf-8")

        frontier = freeze_training_frontier([first_feedback_path, second_feedback_path])
        self.assertEqual(frontier["pareto_protocol_ids"], [second_card["protocol_id"]])
        self.assertEqual(frontier["next_round_parent_protocol_id"], second_card["protocol_id"])
        self.assertFalse(frontier["validation_data_used"])
        self.assertFalse(frontier["test_data_used"])

    def test_validation_run_is_rejected(self) -> None:
        manifest_path = self.run_path.with_suffix(".jsonl.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stage"] = "validation"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stage"):
            build_feedback(
                episode_dir=self.bundle_dir,
                run_path=self.run_path,
                card_path=self.card_path,
                induction_manifest_path=self.induction_path,
                split_seed=self.split_seed,
            )


if __name__ == "__main__":
    unittest.main()
