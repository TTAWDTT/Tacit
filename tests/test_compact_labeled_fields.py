from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.emergent_ood_v0_4.compact_fields import (
    PROTOCOL_ID,
    audit_fields,
    audit_result_rows,
    encode_fields,
)
from experiments.emergent_ood_v0_4.split import build_split
from research.score_compact_fields import (
    DEFAULT_CARD,
    EXPECTED_CARD_SHA256,
    build_report,
    main as score_main,
    verify_run_manifest,
)


TARGET = {
    "shape": "circle",
    "color": "red",
    "quantity": "one",
    "texture": "smooth",
}
CANONICAL = "color=red;quantity=one;shape=circle;texture=smooth"


def result_row(episode_id: str, message: str, *, exact_selection: bool = True) -> dict:
    return {
        "condition": "shared_protocol_card",
        "protocol_id": PROTOCOL_ID,
        "episode_id": episode_id,
        "stage": "test",
        "trace": {
            "message": message,
            "target_tuple_for_evaluator": dict(TARGET),
        },
        "outcome": {"exact_selection": exact_selection},
    }


class CompactLabeledFieldsTests(unittest.TestCase):
    def test_generic_card_matches_pinned_digest(self) -> None:
        digest = hashlib.sha256(Path(DEFAULT_CARD).read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CARD_SHA256)

    def test_encoder_emits_exact_canonical_payload(self) -> None:
        self.assertEqual(encode_fields(TARGET), CANONICAL)

    def test_encoder_rejects_missing_fields_or_reserved_delimiters(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two attributes"):
            encode_fields({"shape": "circle"})
        invalid = dict(TARGET, color="red;quantity=two")
        with self.assertRaisesRegex(ValueError, "grammar delimiters"):
            encode_fields(invalid)
        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            encode_fields(dict(TARGET, color=" red"))

    def test_internal_spaces_in_semantic_fields_are_preserved(self) -> None:
        meaning = {"first name": "Ada Lovelace", "role": "early researcher"}
        payload = encode_fields(meaning)
        self.assertEqual(payload, "first name=Ada Lovelace;role=early researcher")
        audit = audit_fields(payload, meaning)
        self.assertTrue(audit["exact_format_valid"])
        self.assertTrue(audit["canonical_label_fidelity"])

    def test_valid_syntax_parse_and_fidelity_are_distinct_flags(self) -> None:
        self.assertEqual(audit_fields(CANONICAL, TARGET), {
            "syntactic_parse_valid": True,
            "exact_format_valid": True,
            "semantic_parse_valid": True,
            "canonical_label_fidelity": True,
            "failure_reason": None,
        })
        wrong_value = CANONICAL.replace("color=red", "color=blue")
        result = audit_fields(wrong_value, TARGET)
        self.assertTrue(result["exact_format_valid"])
        self.assertTrue(result["semantic_parse_valid"])
        self.assertFalse(result["canonical_label_fidelity"])

    def test_noncanonical_format_and_unknown_labels_fail_separately(self) -> None:
        for message in (
            CANONICAL + ";extra=value",
            CANONICAL.replace("color=red", "color=red "),
            CANONICAL.replace("color=red", "red=red"),
            CANONICAL.replace("color=red;quantity=one", "quantity=one;color=red"),
            CANONICAL.replace("color=red", "color=red=blue"),
        ):
            with self.subTest(message=message):
                self.assertFalse(audit_fields(message, TARGET)["exact_format_valid"])
        reordered = CANONICAL.replace(
            "color=red;quantity=one", "quantity=one;color=red"
        )
        reordered_audit = audit_fields(reordered, TARGET)
        self.assertTrue(reordered_audit["syntactic_parse_valid"])
        self.assertTrue(reordered_audit["semantic_parse_valid"])
        self.assertFalse(reordered_audit["exact_format_valid"])
        self.assertTrue(reordered_audit["canonical_label_fidelity"])
        unknown_label = CANONICAL.replace("color=red", "red=red")
        unknown_label_audit = audit_fields(unknown_label, TARGET)
        self.assertTrue(unknown_label_audit["syntactic_parse_valid"])
        self.assertFalse(unknown_label_audit["semantic_parse_valid"])
        unknown = CANONICAL.replace("color=red", "color=violet")
        result = audit_fields(unknown, TARGET)
        self.assertTrue(result["syntactic_parse_valid"])
        self.assertTrue(result["exact_format_valid"])
        self.assertTrue(result["semantic_parse_valid"])
        self.assertFalse(result["canonical_label_fidelity"])

    def test_same_frozen_grammar_round_trips_all_included_ontologies(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        ontology_paths = [
            project_root / "experiments" / "emergent_ood_v0_4" / "ontologies" / "robotics_v1.json",
            project_root / "experiments" / "emergent_ood_v0_4" / "ontologies" / "music_v1.json",
        ]
        default = {
            "attributes": ["shape", "color", "quantity", "texture"],
            "values_by_attribute": {
                "shape": ["circle", "square", "triangle", "hexagon"],
                "color": ["red", "blue", "green", "yellow"],
                "quantity": ["one", "two", "three", "four"],
                "texture": ["smooth", "rough", "striped", "dotted"],
            },
        }
        specs = [default] + [json.loads(path.read_text(encoding="utf-8")) for path in ontology_paths]
        for spec in specs:
            attributes = spec["attributes"]
            split = build_split(
                seed=17,
                attributes=attributes,
                values=[spec["values_by_attribute"][attribute] for attribute in attributes],
                ontology_id=spec.get("ontology_id"),
            )
            rows = [row for row in split["meanings"] if row["split"] == "held_out"]
            self.assertEqual(len(rows), 64)
            for row in rows:
                meaning = dict(zip(attributes, row["values"]))
                payload = encode_fields(meaning)
                audit = audit_fields(payload, meaning)
                self.assertTrue(all(audit[name] for name in (
                    "syntactic_parse_valid",
                    "exact_format_valid",
                    "semantic_parse_valid",
                    "canonical_label_fidelity",
                )))

    def test_result_audit_preserves_task_success_separately_from_fidelity(self) -> None:
        rows = [
            result_row("episode-1", CANONICAL, exact_selection=True),
            result_row(
                "episode-2", CANONICAL.replace("color=red", "color=blue"),
                exact_selection=True,
            ),
        ]
        audited = audit_result_rows(rows)
        self.assertEqual([row["canonical_label_fidelity"] for row in audited], [True, False])
        self.assertEqual([row["exact_selection"] for row in audited], [True, True])

    def test_report_counts_parser_fidelity_and_task_outcomes_independently(self) -> None:
        rows = [
            result_row("episode-1", CANONICAL, exact_selection=True),
            result_row("episode-2", "malformed", exact_selection=False),
        ]
        report = build_report(
            rows,
            input_sha256="a" * 64,
            card_sha256="b" * 64,
            run_manifest_sha256="c" * 64,
        )
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["summary"], {
            "syntactic_parse_valid_count": 1,
            "exact_format_valid_count": 1,
            "semantic_parse_valid_count": 1,
            "canonical_label_fidelity_count": 1,
            "exact_selection_count": 1,
            "fidelity_and_exact_selection_count": 1,
        })

    def test_manifest_must_bind_single_condition_source_and_pinned_card(self) -> None:
        manifest = {
            "schema": "tlu.emergent-ood-run-manifest.v0.4",
            "results_file": "raw.jsonl",
            "results_sha256": "a" * 64,
            "result_records": 1,
            "conditions": ["shared_protocol_card"],
            "protocol_card_sha256": EXPECTED_CARD_SHA256,
        }
        raw = json.dumps(manifest).encode("utf-8")
        digest = verify_run_manifest(
            raw,
            input_path=Path("raw.jsonl"),
            input_sha256="a" * 64,
            card_sha256=EXPECTED_CARD_SHA256,
            row_count=1,
        )
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        manifest["conditions"] = ["shared_protocol_card", "json"]
        with self.assertRaisesRegex(ValueError, "does not bind"):
            verify_run_manifest(
                json.dumps(manifest).encode("utf-8"),
                input_path=Path("raw.jsonl"),
                input_sha256="a" * 64,
                card_sha256=EXPECTED_CARD_SHA256,
                row_count=1,
            )

    def test_cli_audits_manifest_bound_raw_jsonl_without_overwriting_it(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cache_root = project_root / ".cache"
        cache_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compact-fields-audit-", dir=cache_root) as temp_name:
            temp_root = Path(temp_name).resolve()
            self.assertTrue(temp_root.is_relative_to(project_root.resolve()))
            input_path = temp_root / "raw.jsonl"
            output_path = temp_root / "audit.json"
            raw = (json.dumps(result_row("episode-1", CANONICAL), sort_keys=True) + "\n").encode()
            input_path.write_bytes(raw)
            manifest = {
                "schema": "tlu.emergent-ood-run-manifest.v0.4",
                "results_file": input_path.name,
                "results_sha256": hashlib.sha256(raw).hexdigest(),
                "result_records": 1,
                "conditions": ["shared_protocol_card"],
                "protocol_card_sha256": EXPECTED_CARD_SHA256,
            }
            input_path.with_suffix(input_path.suffix + ".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            stdout = io.StringIO()
            with patch.object(sys, "argv", [
                "score_compact_fields.py", "--input", str(input_path), "--output", str(output_path),
            ]), contextlib.redirect_stdout(stdout):
                self.assertEqual(score_main(), 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["canonical_label_fidelity_count"], 1)
            self.assertEqual(hashlib.sha256(input_path.read_bytes()).hexdigest(), manifest["results_sha256"])

            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "score_compact_fields.py", "--input", str(input_path), "--output", str(input_path), "--force",
            ]), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                score_main()
            self.assertEqual(hashlib.sha256(input_path.read_bytes()).hexdigest(), manifest["results_sha256"])

    def test_result_audit_rejects_wrong_card_and_duplicate_episode(self) -> None:
        wrong_card = result_row("episode-1", CANONICAL)
        wrong_card["protocol_id"] = "other-card"
        with self.assertRaisesRegex(ValueError, "frozen compact-fields"):
            audit_result_rows([wrong_card])
        with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
            audit_result_rows([
                result_row("episode-1", CANONICAL),
                result_row("episode-1", CANONICAL),
            ])


if __name__ == "__main__":
    unittest.main()
