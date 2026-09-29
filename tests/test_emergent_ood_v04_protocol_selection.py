from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.emergent_ood_v0_4.episodes import generate_ledgers, write_ledgers
from experiments.emergent_ood_v0_4.select_protocol_frontier import (
    SPEC_SCHEMA,
    freeze_protocol_frontier,
)
from experiments.emergent_ood_v0_4.induce_protocol_cards import OUTPUT_SCHEMA as INDUCER_SCHEMA
from experiments.emergent_ood_v0_4.split import build_split


class EmergentOODProtocolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cache_root = Path(__file__).resolve().parents[1] / ".cache"
        self.cache_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.cache_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split_seed = 23
        self.task_seed = 7
        self.task_key = bytes(range(32))
        self.bundle = generate_ledgers(
            split=build_split(seed=self.split_seed), task_key=self.task_key,
            task_seed=self.task_seed, k=4, sets_per_stage=2,
        )
        self.bundle_dir = self.root / "episodes"
        write_ledgers(self.bundle, self.bundle_dir)
        self.input_manifest_sha256 = hashlib.sha256((self.bundle_dir / "manifest.json").read_bytes()).hexdigest()

    def _write_candidate(self, protocol_id: str, *, successes: int, wire_bytes: int, tokens: int | None) -> dict[str, str]:
        card_path = self.root / f"{protocol_id}.card.json"
        card = {
            "schema": "tlu.shared_protocol_card.v1",
            "protocol_id": protocol_id,
            "sender_instruction": f"Encode the tuple with protocol {protocol_id}.",
            "receiver_instruction": f"Decode the tuple with protocol {protocol_id}.",
        }
        card_bytes = (json.dumps(card, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        card_path.write_bytes(card_bytes)
        card_digest = hashlib.sha256(card_bytes).hexdigest()

        rows = []
        for index, gold in enumerate(self.bundle["gold"]["validation"]):
            rows.append({
                "schema_version": "tlu.costs.v3",
                "schema": "tlu.emergent-ood-run.v0.4",
                "experiment_id": "emergent-ood-v0.4-receiver-utility",
                "stage": "validation",
                "split_seed": self.split_seed,
                "task_seed": self.task_seed,
                "episode_id": gold["episode_id"],
                "candidate_set_id": gold["candidate_set_id"],
                "condition": "shared_protocol_card",
                "communication_budget_bytes": 4096,
                "protocol_id": protocol_id,
                "protocol": {"policy_id": "shared_protocol_card", "code_id": protocol_id},
                "outcome": {"exact_selection": index < successes},
                "costs": {"application_wire_bytes": wire_bytes},
                "model_calls": [
                    {
                        "agent": "sender", "model": "sender-model", "tokenizer": "sender-tokenizer",
                        "input_tokens": tokens, "output_tokens": 5, "service_seconds": 0.1,
                    },
                    {
                        "agent": "receiver", "model": "receiver-model", "tokenizer": "receiver-tokenizer",
                        "input_tokens": tokens, "output_tokens": 4, "service_seconds": 0.2,
                    },
                ],
                "stratum": {
                    "model_population_id": "same-population",
                    "task_id": "four-attribute-higher-order-meaning-matching-v1",
                    "scorer_id": "strict-candidate-id-and-canonical-sender-fidelity-v1",
                },
            })
        ledger_path = self.root / f"{protocol_id}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        ledger_path.write_bytes(payload)
        manifest = {
            "schema": "tlu.emergent-ood-run-manifest.v0.4",
            "experiment_id": "emergent-ood-v0.4-receiver-utility",
            "stage": "validation",
            "conditions": ["shared_protocol_card"],
            "candidate_sets": 2,
            "candidate_set_offset": 0,
            "communication_budget_bytes": 4096,
            "split_seed": self.split_seed,
            "split_sha256": self.bundle["manifest"]["split_sha256"],
            "task_seed": self.task_seed,
            "task_key_id": self.bundle["manifest"]["task_key_id"],
            "input_episode_manifest_sha256": self.input_manifest_sha256,
            "protocol_card_sha256": card_digest,
            "sender_model": "sender-model",
            "receiver_model": "receiver-model",
            "sender_tokenizer_id": "sender-tokenizer",
            "receiver_tokenizer_id": "receiver-tokenizer",
            "model_population_id": "same-population",
            "results_file": ledger_path.name,
            "results_sha256": hashlib.sha256(payload).hexdigest(),
            "result_records": len(rows),
        }
        ledger_path.with_suffix(ledger_path.suffix + ".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return {
            "protocol_id": protocol_id, "card": card_path.name,
            "card_sha256": card_digest, "validation_ledgers": [ledger_path.name],
        }

    def _write_induction_manifest(self, candidates: list[dict[str, str]]) -> str:
        prompt_path = self.root / "induction-prompt.json"
        completion_path = self.root / "induction-completion.txt"
        prompt = b"{\"messages\": []}"
        completion = b"{\"candidate_cards\": []}"
        prompt_path.write_bytes(prompt)
        completion_path.write_bytes(completion)
        project_root = Path(__file__).resolve().parents[1]
        manifest_path = self.root / "induction-manifest.json"
        manifest = {
            "schema": INDUCER_SCHEMA,
            "mode": "execute",
            "inference_started": True,
            "model_calls": 1,
            "split_seed": self.split_seed,
            "split_sha256": self.bundle["manifest"]["split_sha256"],
            "task_key_id": self.bundle["manifest"]["task_key_id"],
            "protocol_family": "compositional_symbolic",
            "inducer_model": "generator-model",
            "tokenizer_id": "generator-tokenizer",
            "input_tokens": 123,
            "output_tokens": 45,
            "service_seconds": 1.2,
            "prompt_path": prompt_path.relative_to(project_root).as_posix(),
            "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
            "prompt_utf8_bytes": len(prompt),
            "completion_path": completion_path.relative_to(project_root).as_posix(),
            "completion_sha256": hashlib.sha256(completion).hexdigest(),
            "completion_utf8_bytes": len(completion),
            "candidate_cards": [
                {
                    "protocol_id": candidate["protocol_id"],
                    "path": (self.root / candidate["card"]).relative_to(project_root).as_posix(),
                    "sha256": candidate["card_sha256"],
                    "bytes": (self.root / candidate["card"]).stat().st_size,
                }
                for candidate in candidates
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path.name

    def test_freezes_paired_validation_pareto_set_and_never_opens_test_roles(self):
        candidates = [
            self._write_candidate("accurate-costly", successes=8, wire_bytes=180, tokens=100),
            self._write_candidate("compact-lower-accuracy", successes=6, wire_bytes=80, tokens=40),
            self._write_candidate("dominated", successes=6, wire_bytes=100, tokens=60),
        ]
        spec = {
            "schema": SPEC_SCHEMA,
            "input_dir": "episodes",
            "split_seed": self.split_seed,
            "reuse_horizon_evaluation_episodes": 1000,
            "candidates": candidates,
        }
        spec_path = self.root / "candidates.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        # The manifest still declares test files, but this freeze operation must not read them.
        for name in ("sender_test.jsonl", "receiver_test.jsonl", "gold_test.jsonl"):
            (self.bundle_dir / name).unlink()
        report = freeze_protocol_frontier(spec_path)

        self.assertEqual(report["pareto_protocol_ids"], ["accurate-costly", "compact-lower-accuracy"])
        self.assertEqual(report["validation_episode_count"], 8)
        self.assertFalse(report["evaluation_data_used"])
        self.assertFalse(report["test_stage_files_opened"])
        self.assertEqual(report["selection_setup_cost"]["validation_candidate_model_calls"], 48)
        self.assertEqual(report["selection_setup_cost"]["model_input_tokens_by_agent_model_tokenizer"]["sender|sender-model|sender-tokenizer"], 1600)
        self.assertEqual(report["selection_setup_cost"]["protocol_induction"]["status"], "not_supplied")

    def test_induction_costs_and_card_hashes_are_bound_into_freeze(self):
        candidates = [
            self._write_candidate("accurate-costly", successes=8, wire_bytes=180, tokens=100),
            self._write_candidate("compact-lower-accuracy", successes=6, wire_bytes=80, tokens=40),
        ]
        induction_manifest = self._write_induction_manifest(candidates)
        spec_path = self.root / "candidates.json"
        spec_path.write_text(json.dumps({
            "schema": SPEC_SCHEMA,
            "input_dir": "episodes",
            "split_seed": self.split_seed,
            "reuse_horizon_evaluation_episodes": 100,
            "induction_manifests": [induction_manifest],
            "candidates": candidates,
        }), encoding="utf-8")

        report = freeze_protocol_frontier(spec_path)
        induction = report["selection_setup_cost"]["protocol_induction"]
        self.assertEqual(induction["status"], "verified_induction_manifests")
        self.assertEqual(induction["model_calls"], 1)
        self.assertEqual(induction["input_tokens_by_model_tokenizer"], {
            "protocol_inducer|compositional_symbolic|generator-model|generator-tokenizer": 123,
        })
        self.assertEqual(induction["prompt_completion_utf8_bytes"], 39)
        self.assertEqual(induction["candidate_protocol_families"], {
            "accurate-costly": "compositional_symbolic",
            "compact-lower-accuracy": "compositional_symbolic",
        })
        amortized = report["selection_setup_cost"]["amortized_protocol_induction_cost_per_reuse_episode"]
        self.assertAlmostEqual(amortized["model_calls"], 0.01)
        self.assertEqual(amortized["model_output_tokens_by_model_tokenizer"], {
            "protocol_inducer|compositional_symbolic|generator-model|generator-tokenizer": 0.45,
        })

    def test_rejects_incomplete_or_repeated_validation_batch_coverage(self):
        candidate = self._write_candidate("candidate-a", successes=8, wire_bytes=100, tokens=50)
        second = self._write_candidate("candidate-b", successes=7, wire_bytes=110, tokens=60)
        # A duplicated batch range cannot masquerade as coverage of the full validation stage.
        second_manifest_path = self.root / "candidate-b.jsonl.manifest.json"
        manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
        manifest["candidate_set_offset"] = 1
        second_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        spec_path = self.root / "candidates.json"
        spec_path.write_text(json.dumps({
            "schema": SPEC_SCHEMA,
            "input_dir": "episodes",
            "split_seed": self.split_seed,
            "reuse_horizon_evaluation_episodes": 100,
            "candidates": [candidate, second],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "results hash|overlap|omit"):
            freeze_protocol_frontier(spec_path)

    def test_missing_token_usage_removes_token_axes_instead_of_treating_them_as_zero(self):
        candidates = [
            self._write_candidate("measured", successes=8, wire_bytes=100, tokens=40),
            self._write_candidate("missing-usage", successes=7, wire_bytes=110, tokens=None),
        ]
        spec_path = self.root / "candidates.json"
        spec_path.write_text(json.dumps({
            "schema": SPEC_SCHEMA,
            "input_dir": "episodes",
            "split_seed": self.split_seed,
            "reuse_horizon_evaluation_episodes": 100,
            "candidates": candidates,
        }), encoding="utf-8")
        report = freeze_protocol_frontier(spec_path)
        self.assertEqual(report["pareto_protocol_ids"], ["measured"])
        self.assertEqual(report["pareto_dimensions"], [
            "validation_success_rate|max", "application_wire_bytes_per_episode|min",
        ])
        usage = {candidate["protocol_id"]: candidate["token_usage_complete"] for candidate in report["candidate_metrics"]}
        self.assertEqual(usage, {"measured": True, "missing-usage": False})
        self.assertIsNone(report["selection_setup_cost"]["model_input_tokens_by_agent_model_tokenizer"])


if __name__ == "__main__":
    unittest.main()
