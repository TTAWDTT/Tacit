from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.emergent_ood_v0_4.episodes import generate_ledgers, write_ledgers
import experiments.emergent_ood_v0_4.generate_usage_examples as generator
import experiments.emergent_ood_v0_4.shuffle_usage_examples as shuffler
from experiments.emergent_ood_v0_4.runner import load_episode_bundle, load_protocol_card, load_usage_examples
from experiments.emergent_ood_v0_4.split import build_split
from tacit.runtime import ChatCompletion


class FakeSender:
    def __init__(self):
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        request = json.loads(messages[-1]["content"])
        meaning = request["private_meaning"]
        return ChatCompletion(
            json.dumps({"message": json.dumps(meaning, ensure_ascii=False, separators=(",", ":"))}),
            "fake-sender", 19, 17, 0.02,
        )


class InterruptAfterOneSender:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated request interruption")
        return FakeSender().complete(messages)


class InvalidThenGoodSender:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return ChatCompletion("not valid JSON", "fake-sender", 11, 2, 0.03)
        return FakeSender().complete(messages)


class SuccessThenInterruptSender:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated request interruption")
        return FakeSender().complete(messages)


class UsageExampleGenerationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.cache_root = self.root / ".cache"
        self.cache_root.mkdir(exist_ok=True)
        self.split = build_split(seed=17)
        self.bundle = generate_ledgers(
            split=self.split, task_key=bytes(range(32)), task_seed=9, k=4, sets_per_stage=3
        )

    def _inputs(self, root: Path):
        bundle_dir = root / "episodes"
        write_ledgers(self.bundle, bundle_dir)
        key_path = root / "evaluator.key"
        key_path.write_bytes(bytes(range(32)))
        card_path = root / "card.json"
        card_path.write_text(json.dumps({
            "schema": "tlu.shared_protocol_card.v1",
            "protocol_id": "generation-test-card-v1",
            "sender_instruction": "Return the complete tuple as compact JSON.",
            "receiver_instruction": "Decode the tuple.",
        }), encoding="utf-8")
        preflight = root / "preflight.json"
        preflight.write_text('{"schema":"test"}\n', encoding="utf-8")
        return bundle_dir, key_path, card_path, preflight

    def test_training_selection_is_deterministic_unique_and_labeled_only_by_private_artifact(self):
        first = generator.select_training_meanings(
            bundle=self.bundle, split=self.split, task_key=bytes(range(32)), count=4,
        )
        second = generator.select_training_meanings(
            bundle=self.bundle, split=self.split, task_key=bytes(range(32)), count=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(len({row["meaning_id"] for row in first}), 4)
        self.assertTrue(set(row["meaning_id"] for row in first).issubset(self.split["train_meaning_ids"]))
        self.assertEqual(set(first[0]), {"meaning_id", "meaning"})
        with self.assertRaisesRegex(ValueError, "1..12"):
            generator.select_training_meanings(
                bundle=self.bundle, split=self.split, task_key=bytes(range(32)), count=13,
            )

    def test_dry_run_plans_without_creating_artifacts_or_constructing_a_client(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, _ = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "OpenAICompatibleClient") as client:
                plan = generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=False,
                )
            client.assert_not_called()
            self.assertEqual(plan["planned_model_calls"], 3)
            self.assertFalse(plan["inference_started"])
            self.assertFalse(output_dir.exists())

    def test_execute_writes_bound_artifact_and_real_cost_ledger(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            fake_sender = FakeSender()
            with patch.object(generator, "OpenAICompatibleClient", return_value=fake_sender), patch.object(
                generator, "validate_resource_preflight"
            ) as gate:
                manifest = generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=True,
                    resource_preflight=preflight,
                )
            gate.assert_called_once_with(preflight.resolve(), required_ports={8000})
            artifact = json.loads((output_dir / "usage-examples.json").read_text(encoding="utf-8"))
            self.assertEqual(len(artifact["examples"]), 3)
            self.assertEqual(artifact["acquisition"]["generation_calls"], 3)
            self.assertEqual(artifact["acquisition"]["input_tokens"], 57)
            self.assertEqual(artifact["acquisition"]["output_tokens"], 51)
            self.assertEqual(manifest["model_calls"], 3)
            self.assertEqual(len(manifest["example_traces"]), 3)
            saved_trace = [
                json.loads(line)
                for line in (output_dir / "generation-trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved_trace), 3)
            self.assertIn("raw_completion", saved_trace[0])
            self.assertEqual(manifest["generation_trace_sha256"], hashlib.sha256(
                (output_dir / "generation-trace.jsonl").read_bytes()
            ).hexdigest())
            self.assertEqual(len(fake_sender.prompts), 3)
            for prompt in fake_sender.prompts:
                combined_prompt = "\n".join(message["content"] for message in prompt)
                self.assertIn("private_meaning", combined_prompt)
                self.assertNotIn("meaning_id", combined_prompt)
                self.assertNotIn("candidate_id", combined_prompt)
                self.assertNotIn("validation", combined_prompt)
            loaded, split = load_episode_bundle(bundle_dir, split_seed=17)
            card, card_hash = load_protocol_card(card_path)
            examples, digest, metadata = load_usage_examples(
                output_dir / "usage-examples.json", split=split, bundle=loaded,
                protocol_card=card, protocol_card_sha256=card_hash,
                training_episode_manifest_sha256=hashlib.sha256((bundle_dir / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(examples), 3)
            self.assertEqual(metadata["acquisition"]["tokenizer_id"], "fake-tokenizer")

    def test_shuffled_pair_control_is_deterministic_bound_and_still_runner_loadable(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "OpenAICompatibleClient", return_value=FakeSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=4, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=True,
                    resource_preflight=preflight,
                )
            source_path = output_dir / "usage-examples.json"
            first_path = root / "shuffled-a.json"
            second_path = root / "shuffled-b.json"
            first_manifest = shuffler.shuffle_usage_examples(
                source_path=source_path, output_path=first_path, seed=31,
            )
            second_manifest = shuffler.shuffle_usage_examples(
                source_path=source_path, output_path=second_path, seed=31,
            )
            original = json.loads(source_path.read_text(encoding="utf-8"))
            shuffled = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                sorted(row["message"] for row in shuffled["examples"]),
                sorted(row["message"] for row in original["examples"]),
            )
            self.assertTrue(all(
                row["message"] != original["examples"][index]["message"]
                for index, row in enumerate(shuffled["examples"])
            ))
            self.assertEqual(shuffled["acquisition"], original["acquisition"])
            self.assertEqual(first_manifest["model_calls_added"], 0)

            bundle, split = load_episode_bundle(bundle_dir, split_seed=17)
            card, card_hash = load_protocol_card(card_path)
            loaded, digest, _ = load_usage_examples(
                first_path, split=split, bundle=bundle, protocol_card=card,
                protocol_card_sha256=card_hash,
                training_episode_manifest_sha256=hashlib.sha256(
                    (bundle_dir / "manifest.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(len(loaded), 4)
            self.assertEqual(digest, first_manifest["control_artifact_sha256"])

    def test_shuffled_pair_control_rejects_unshuffleable_labels_and_existing_output(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            source_path = root / "source.json"
            source = {
                "schema": "tlu.usage_examples.v1",
                "examples": [
                    {"meaning_id": "a", "meaning": {"x": "1"}, "message": "same"},
                    {"meaning_id": "b", "meaning": {"x": "2"}, "message": "same"},
                ],
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not permit"):
                shuffler.shuffle_usage_examples(
                    source_path=source_path, output_path=root / "unshuffleable.json", seed=0,
                )
            source["examples"][1]["message"] = "different"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            output_path = root / "control.json"
            output_path.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "outputs must be new"):
                shuffler.shuffle_usage_examples(
                    source_path=source_path, output_path=output_path, seed=0,
                )

    def test_interruption_checkpoints_exact_prefix_and_resume_reuses_it(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "OpenAICompatibleClient", return_value=InterruptAfterOneSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                with self.assertRaisesRegex(RuntimeError, "attempt and checkpoint were saved"):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight,
                    )
            checkpoint = json.loads((output_dir / "generation.checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(len(checkpoint["results"]), 1)
            with patch.object(generator, "OpenAICompatibleClient", return_value=FakeSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                resumed = generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=True,
                    resource_preflight=preflight, resume=True,
                )
            self.assertEqual(resumed["model_calls"], 4)
            self.assertEqual(resumed["failed_attempts"], 1)
            self.assertFalse((output_dir / "generation.checkpoint.json").exists())

    def test_invalid_completion_retry_is_preserved_and_charged_in_usage_artifact(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "OpenAICompatibleClient", return_value=InvalidThenGoodSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                with self.assertRaisesRegex(ValueError, "failed attempt and checkpoint were saved"):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=1, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight,
                    )
            with patch.object(generator, "OpenAICompatibleClient", return_value=FakeSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                manifest = generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=1, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=True,
                    resource_preflight=preflight, resume=True,
                )
            artifact = json.loads((output_dir / "usage-examples.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["acquisition"]["generation_calls"], 2)
            self.assertEqual(artifact["acquisition"]["input_tokens"], 30)
            self.assertEqual(artifact["acquisition"]["output_tokens"], 19)
            self.assertEqual(manifest["failed_attempts"], 1)
            trace = [
                json.loads(line)
                for line in (output_dir / "generation-trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in trace], ["failed", "success"])

    def test_resume_validates_old_failure_after_successful_prefix_advances(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "OpenAICompatibleClient", return_value=InvalidThenGoodSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                with self.assertRaises(ValueError):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight,
                    )
            with patch.object(generator, "OpenAICompatibleClient", return_value=SuccessThenInterruptSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                with self.assertRaisesRegex(RuntimeError, "attempt and checkpoint were saved"):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight, resume=True,
                    )
            with patch.object(generator, "OpenAICompatibleClient", return_value=FakeSender()), patch.object(
                generator, "validate_resource_preflight"
            ):
                manifest = generator.generate_usage_examples(
                    input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                    protocol_card_path=card_path, example_count=3, output_dir=output_dir,
                    model="fake-sender", tokenizer_id="fake-tokenizer",
                    base_url="http://127.0.0.1:8000/v1", execute=True,
                    resource_preflight=preflight, resume=True,
                )
            artifact = json.loads((output_dir / "usage-examples.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["acquisition"]["generation_calls"], 5)
            self.assertIsNone(artifact["acquisition"]["input_tokens"])
            self.assertEqual(manifest["failed_attempts"], 2)
            trace = [
                json.loads(line)
                for line in (output_dir / "generation-trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in trace], ["failed", "success", "failed", "success", "success"])

    def test_cumulative_attempt_cap_stops_retries_before_an_extra_request(self):
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temporary:
            root = Path(temporary)
            bundle_dir, key_path, card_path, preflight = self._inputs(root)
            output_dir = root / "generated"
            with patch.object(generator, "MAX_CALLS_PER_BATCH", 1), patch.object(
                generator, "OpenAICompatibleClient", return_value=InvalidThenGoodSender()
            ), patch.object(generator, "validate_resource_preflight"):
                with self.assertRaises(ValueError):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=1, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight,
                    )
            retry_client = FakeSender()
            with patch.object(generator, "MAX_CALLS_PER_BATCH", 1), patch.object(
                generator, "OpenAICompatibleClient", return_value=retry_client
            ), patch.object(generator, "validate_resource_preflight"):
                with self.assertRaisesRegex(RuntimeError, "exhausted its 1 total request attempts"):
                    generator.generate_usage_examples(
                        input_dir=bundle_dir, split_seed=17, task_key_path=key_path,
                        protocol_card_path=card_path, example_count=1, output_dir=output_dir,
                        model="fake-sender", tokenizer_id="fake-tokenizer",
                        base_url="http://127.0.0.1:8000/v1", execute=True,
                        resource_preflight=preflight, resume=True,
                    )
            self.assertEqual(retry_client.prompts, [])


if __name__ == "__main__":
    unittest.main()
