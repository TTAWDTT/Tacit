from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.emergent_ood_v0_4.split import build_split
from experiments.emergent_ood_v0_4.compact_fields import encode_fields as encode_v3_fields
from research.quoted_labeled_fields import (
    audit_fields,
    compare_payload_bytes,
    decode_fields,
    encode_fields,
)


class QuotedLabeledFieldsTests(unittest.TestCase):
    def test_round_trips_delimiters_quotes_backslashes_controls_and_unicode(self) -> None:
        meaning = {
            "key;=\\\"": "value;=\\\"",
            "line\nbreak": "tab\treturn\rnull\u0000",
            "音符": "𝄞 café",
        }
        payload = encode_fields(meaning)
        self.assertEqual(decode_fields(payload), meaning)
        self.assertTrue(all(audit_fields(payload, meaning)[key] for key in (
            "syntactic_parse_valid",
            "exact_format_valid",
            "semantic_parse_valid",
            "canonical_label_fidelity",
        )))

    def test_compact_json_object_is_exactly_two_utf8_bytes_longer(self) -> None:
        meaning = {"alpha": "one", "beta": "two"}
        self.assertEqual(compare_payload_bytes(meaning), {
            "quoted_labeled_fields_bytes": len(encode_fields(meaning).encode("utf-8")),
            "compact_json_bytes": len(json.dumps(
                meaning, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")),
            "bytes_saved_vs_json": 2,
        })

    def test_v3_punctuation_saving_is_four_quotes_per_field_plus_object_braces(self) -> None:
        from experiments.emergent_ood_v0_4.compact_fields import encode_fields as encode_v3

        for field_count in range(2, 9):
            meaning = {
                f"key{index}": f"value{index}"
                for index in range(field_count)
            }
            json_bytes = compare_payload_bytes(meaning)["compact_json_bytes"]
            self.assertEqual(
                json_bytes - len(encode_v3(meaning).encode("utf-8")),
                4 * field_count + 2,
            )

    def test_parser_rejects_duplicates_bad_json_and_dangling_separator(self) -> None:
        for malformed in (
            '"a"="1";"a"="2"',
            '"a"="unterminated',
            '"a"="1";',
            '"a"="1";"b"=true',
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                decode_fields(malformed)

    def test_valid_noncanonical_json_escape_is_parsed_but_not_canonical(self) -> None:
        audit = audit_fields(r'"\u0061"="x"', {"a": "x"})
        self.assertTrue(audit["syntactic_parse_valid"])
        self.assertFalse(audit["exact_format_valid"])
        self.assertTrue(audit["semantic_parse_valid"])
        self.assertTrue(audit["canonical_label_fidelity"])

    def test_unpaired_surrogates_are_rejected_for_utf8_interoperability(self) -> None:
        with self.assertRaisesRegex(ValueError, "unpaired surrogate"):
            encode_fields({"key": "\ud800"})
        with self.assertRaisesRegex(ValueError, "unpaired surrogate"):
            decode_fields('"key"="\ud800"')

    def test_all_heldout_tuples_in_three_ontologies_save_only_two_bytes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default = {
            "attributes": ["shape", "color", "quantity", "texture"],
            "values_by_attribute": {
                "shape": ["circle", "square", "triangle", "hexagon"],
                "color": ["red", "blue", "green", "yellow"],
                "quantity": ["one", "two", "three", "four"],
                "texture": ["smooth", "rough", "striped", "dotted"],
            },
        }
        specs = [default]
        specs.extend(json.loads(path.read_text(encoding="utf-8")) for path in (
            project_root / "experiments/emergent_ood_v0_4/ontologies/robotics_v1.json",
            project_root / "experiments/emergent_ood_v0_4/ontologies/music_v1.json",
        ))
        audited = 0
        for spec in specs:
            attributes = spec["attributes"]
            split = build_split(
                seed=17,
                attributes=attributes,
                values=[spec["values_by_attribute"][name] for name in attributes],
                ontology_id=spec.get("ontology_id"),
            )
            heldout = [row for row in split["meanings"] if row["split"] == "held_out"]
            self.assertEqual(len(heldout), 64)
            for row in heldout:
                meaning = dict(zip(attributes, row["values"]))
                comparison = compare_payload_bytes(meaning)
                json_bytes = comparison["compact_json_bytes"]
                self.assertEqual(comparison["bytes_saved_vs_json"], 2)
                self.assertEqual(json_bytes - len(encode_v3_fields(meaning).encode("utf-8")), 18)
                self.assertEqual(decode_fields(encode_fields(meaning)), meaning)
                audited += 1
        self.assertEqual(audited, 192)


if __name__ == "__main__":
    unittest.main()
